import os, re, json, sqlite3, uuid
from pathlib import Path
from typing import Optional
import fitz
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
UPLOADS = DATA / "uploads"
DATA.mkdir(exist_ok=True); UPLOADS.mkdir(exist_ok=True)
DB = DATA / "study.db"
load_dotenv(BASE / ".env")

app = FastAPI(title="AI Study Assistant", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

with db() as c:
    c.executescript('''
    CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, name TEXT, path TEXT, text TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS activity(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, value INTEGER DEFAULT 1, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    ''')


def clean_text(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def chunks(text: str, size=1000, overlap=150):
    words = text.split()
    out=[]
    i=0
    while i < len(words):
        part=' '.join(words[i:i+size])
        if part: out.append(part)
        i += max(1, size-overlap)
    return out


def all_chunks():
    with db() as c:
        docs = c.execute('SELECT id,name,text FROM documents ORDER BY created_at DESC').fetchall()
    result=[]
    for d in docs:
        for i,ch in enumerate(chunks(d['text'])):
            result.append({'doc_id':d['id'],'name':d['name'],'chunk_id':i,'text':ch})
    return result


def retrieve(query, k=5):
    items=all_chunks()
    if not items: return []
    corpus=[x['text'] for x in items]
    vec=TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    X=vec.fit_transform(corpus)
    q=vec.transform([query])
    scores=cosine_similarity(q,X).ravel()
    ids=scores.argsort()[::-1][:k]
    return [{**items[i], 'score':float(scores[i])} for i in ids if scores[i] > 0]


def sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 30]


def extractive_summary(text, n=7):
    ss=sentences(text)
    if not ss: return text[:1500]
    if len(ss) <= n: return ' '.join(ss)
    try:
        v=TfidfVectorizer(stop_words='english')
        X=v.fit_transform(ss)
        scores=X.sum(axis=1).A1
        best=sorted(range(len(ss)), key=lambda i:scores[i], reverse=True)[:n]
        return ' '.join(ss[i] for i in sorted(best))
    except Exception:
        return ' '.join(ss[:n])


def fallback_answer(question, context):
    ss=sentences(context)
    qwords=set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
    ranked=[]
    for s in ss:
        words=set(re.findall(r'\b[a-zA-Z]{3,}\b', s.lower()))
        overlap=len(qwords & words)
        ranked.append((overlap,s))
    ranked.sort(reverse=True)
    chosen=[s for score,s in ranked[:5] if score]
    if not chosen: chosen=ss[:4]
    return ' '.join(chosen)[:3500] or 'I could not find enough information in the uploaded material.'


async def llm(prompt):
    key=os.getenv('LLM_API_KEY','').strip()
    base=os.getenv('LLM_BASE_URL','https://api.openai.com/v1').rstrip('/')
    model=os.getenv('LLM_MODEL','gpt-4o-mini')
    if not key: return None
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r=await client.post(base+'/chat/completions', headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'}, json={'model':model,'messages':[{'role':'system','content':'You are a helpful college study assistant. Use only the supplied study context. If context is insufficient, say so.'},{'role':'user','content':prompt}], 'temperature':0.2})
            r.raise_for_status(); return r.json()['choices'][0]['message']['content']
    except Exception:
        return None

class Ask(BaseModel):
    question:str
    document_id:Optional[str]=None

class Gen(BaseModel):
    document_id:Optional[str]=None
    count:int=5

@app.get('/api/health')
def health(): return {'ok':True,'llm_enabled':bool(os.getenv('LLM_API_KEY'))}

@app.get('/api/documents')
def documents():
    with db() as c: rows=c.execute('SELECT id,name,created_at,length(text) as chars FROM documents ORDER BY created_at DESC').fetchall()
    return [dict(r) for r in rows]

@app.post('/api/upload')
async def upload(file:UploadFile=File(...)):
    if not file.filename.lower().endswith(('.pdf','.txt')): raise HTTPException(400,'Only PDF and TXT files are supported.')
    data=await file.read()
    doc_id=str(uuid.uuid4())
    path=UPLOADS/(doc_id+'_'+Path(file.filename).name)
    path.write_bytes(data)
    if file.filename.lower().endswith('.pdf'):
        doc=fitz.open(stream=data, filetype='pdf')
        text='\n'.join(p.get_text() for p in doc)
    else: text=data.decode('utf-8',errors='ignore')
    text=clean_text(text)
    if not text: raise HTTPException(400,'No readable text was found in this file.')
    with db() as c: c.execute('INSERT INTO documents(id,name,path,text) VALUES(?,?,?,?)',(doc_id,file.filename,str(path),text))
    return {'id':doc_id,'name':file.filename,'chars':len(text)}

@app.delete('/api/documents/{doc_id}')
def delete_doc(doc_id:str):
    with db() as c:
        r=c.execute('SELECT path FROM documents WHERE id=?',(doc_id,)).fetchone()
        if not r: raise HTTPException(404,'Document not found')
        c.execute('DELETE FROM documents WHERE id=?',(doc_id,))
    try: Path(r['path']).unlink(missing_ok=True)
    except Exception: pass
    return {'ok':True}

@app.post('/api/ask')
async def ask(body:Ask):
    items=retrieve(body.question, 6)
    if body.document_id: items=[x for x in items if x['doc_id']==body.document_id]
    context='\n\n'.join(f"[{x['name']}] {x['text']}" for x in items)
    if not context: raise HTTPException(400,'Upload a study document first, or ask a question matching its content.')
    prompt=f"Question: {body.question}\n\nContext:\n{context}\n\nAnswer clearly for a college student. Explain briefly and cite the document names in parentheses."
    answer=await llm(prompt) or fallback_answer(body.question, context)
    with db() as c: c.execute("INSERT INTO activity(kind) VALUES('question')")
    return {'answer':answer,'sources':[{'name':x['name'],'snippet':x['text'][:260],'score':round(x['score'],3)} for x in items]}


def selected_text(doc_id=None):
    with db() as c:
        if doc_id:
            r=c.execute('SELECT text,name FROM documents WHERE id=?',(doc_id,)).fetchone()
            return (r['text'],r['name']) if r else ('','')
        rs=c.execute('SELECT text,name FROM documents ORDER BY created_at DESC').fetchall()
    return ('\n\n'.join(r['text'] for r in rs),'All documents')

@app.post('/api/summary')
async def summary(body:Gen):
    text,name=selected_text(body.document_id)
    if not text: raise HTTPException(400,'No document available.')
    context=text[:14000]
    prompt=f"Create a structured study summary of this material. Include: Overview, key concepts, important definitions, and exam points.\nDocument: {name}\nMaterial:\n{context}"
    ans=await llm(prompt) or extractive_summary(text,9)
    with db() as c: c.execute("INSERT INTO activity(kind) VALUES('summary')")
    return {'title':name,'summary':ans}

@app.post('/api/flashcards')
async def flashcards(body:Gen):
    text,name=selected_text(body.document_id)
    if not text: raise HTTPException(400,'No document available.')
    prompt=f"Create {max(1,min(body.count,15))} concise study flashcards from this material. Return ONLY valid JSON array of objects with keys question and answer. Material: {text[:16000]}"
    raw=await llm(prompt)
    cards=[]
    if raw:
        try: cards=json.loads(re.search(r'\[.*\]',raw,re.S).group(0))
        except Exception: pass
    if not cards:
        ss=sentences(text)
        for i,s in enumerate(ss[:body.count]):
            cards.append({'question':f'What is the main idea of this statement?','answer':s})
    with db() as c: c.execute("INSERT INTO activity(kind,value) VALUES('flashcards',?)",(len(cards),))
    return {'cards':cards[:body.count]}

@app.post('/api/mcqs')
async def mcqs(body:Gen):
    text,name=selected_text(body.document_id)
    if not text: raise HTTPException(400,'No document available.')
    count=max(1,min(body.count,15))
    prompt=f"Create {count} college-level multiple-choice questions from this material. Return ONLY valid JSON array. Each object: question, options (array of 4 strings), answer (exact option string), explanation. Material: {text[:18000]}"
    raw=await llm(prompt); qs=[]
    if raw:
        try: qs=json.loads(re.search(r'\[.*\]',raw,re.S).group(0))
        except Exception: pass
    if not qs:
        ss=sentences(text)
        for s in ss[:count]:
            qs.append({'question':'Which statement is supported by the uploaded material?','options':[s,'This is not stated in the material.','The material gives the opposite claim.','None of these.'],'answer':s,'explanation':'This option is taken directly from the study material.'})
    with db() as c: c.execute("INSERT INTO activity(kind,value) VALUES('mcq',?)",(len(qs),))
    return {'questions':qs[:count]}

@app.get('/api/stats')
def stats():
    with db() as c:
        docs=c.execute('SELECT COUNT(*) n FROM documents').fetchone()['n']
        questions=c.execute("SELECT COUNT(*) n FROM activity WHERE kind='question'").fetchone()['n']
        summaries=c.execute("SELECT COUNT(*) n FROM activity WHERE kind='summary'").fetchone()['n']
        cards=c.execute("SELECT COALESCE(SUM(value),0) n FROM activity WHERE kind='flashcards'").fetchone()['n']
        mcqs=c.execute("SELECT COALESCE(SUM(value),0) n FROM activity WHERE kind='mcq'").fetchone()['n']
    return {'documents':docs,'questions':questions,'summaries':summaries,'flashcards':cards,'mcqs':mcqs}
