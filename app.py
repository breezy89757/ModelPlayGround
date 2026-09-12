"""本地 POC 伺服器。跑法： uvicorn app:app --reload --port 8000

狀態全部由前端持有並回傳，伺服器無 session。
"""
import json
import asyncio
import pathlib

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, Response
from pydantic import BaseModel

from core import pipeline, pack, store, calibration

ROOT = pathlib.Path(__file__).resolve().parent
app = FastAPI(title="ModelPlayGround POC")


def sse(worker_factory):
    """把一個 async worker 包成 SSE 串流。worker 收到同步的 emit。"""
    queue: asyncio.Queue = asyncio.Queue()

    def emit(event: str, payload: dict):
        queue.put_nowait((event, payload))

    async def worker():
        try:
            await worker_factory(emit)
        except Exception as e:
            emit("error", {"message": f"{type(e).__name__}: {e}"})
        finally:
            emit("done", {})

    async def gen():
        task = asyncio.create_task(worker())
        try:
            while True:
                event, payload = await queue.get()
                yield f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                if event == "done":
                    break
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/config")
def config():
    return {"model": pipeline.target_model()}


class Ask(BaseModel):
    text: str


@app.post("/api/analyze")
async def analyze(body: Ask):
    return sse(lambda emit: pipeline.run(body.text, emit))


class Rerun(BaseModel):
    system_prompt: str
    cases: list
    run: int = 2
    requirements: dict = {}


@app.post("/api/rerun")
async def rerun(body: Rerun):
    """套用修補後重跑。刻意重跑**全部**情境——修補可能弄壞原本會過的。"""

    async def work(emit):
        results = await pipeline.evaluate(
            body.system_prompt, body.cases, emit, run=body.run)
        v = await pipeline.verdict(body.cases, results, emit)
        pipeline.persist({"requirements": body.requirements, "cases": body.cases,
                          "system_prompt": body.system_prompt, "results": results,
                          "run": body.run, **v}, emit)

    return sse(work)


@app.get("/api/runs")
def runs():
    return {"runs": store.listing()}


@app.get("/api/calibration")
def calibration_report():
    labels = calibration.load_labels()
    rep = calibration.report(labels)
    return {"n_labels": len(labels), "report": rep.dict() if rep else None}


class PatchReq(BaseModel):
    system_prompt: str
    failures: list


@app.post("/api/patch")
async def patch(body: PatchReq):
    if not body.failures:
        return {"diagnosis": "沒有失敗案例", "patch": "", "addresses": []}
    return await pipeline.suggest_patch(body.system_prompt, body.failures)


class ExportReq(BaseModel):
    requirements: dict = {}
    cases: list = []
    system_prompt: str = ""
    stats: dict = {}
    verdict: dict = {}


@app.post("/api/export")
async def export(body: ExportReq):
    state = body.model_dump()
    doc = await pack.skill_doc(state)
    name, blob = pack.build(state, doc)
    return Response(
        content=blob, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
