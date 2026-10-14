from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
import os
import time
from main_chunking import ChunkData
import structlog
import logging
from typing import Optional

OUTPUT_PATH = "/tmp/output_ocr"

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True
)
logger = structlog.get_logger(service="chunking_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the FastAPI app"""
    # Startup
    app.state.semantic_chunking_lock = asyncio.Lock()
    print("🚀 Starting chunking API...")
    print("✅ API startup complete")
    
    yield
    print("🛑 Shutting down chunking API...")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/services/chunking/doc")
async def process_(      
    request: Request,
    content_parsed: str = Form(...),
    doc_id: Optional[str] = Form(None),
    min_split_tokens: int = Form(150, ge=1),
    max_split_tokens: int = Form(300, ge=1),
    split_tokens_tolerance: int = Form(10, ge=0),
    window_size: int = Form(5, ge=1),
):
    if min_split_tokens >= max_split_tokens:
        raise HTTPException(
            status_code=422,
            detail="min_split_tokens must be smaller than max_split_tokens",
        )

    start_time = time.time()
    logger.info(
        event="chunking_start",
        endpoint="doc",
        doc_id=doc_id,
        min_split_tokens=min_split_tokens,
        max_split_tokens=max_split_tokens,
        split_tokens_tolerance=split_tokens_tolerance,
        window_size=window_size,
    )
    try:
        chunking = ChunkData()
        async with request.app.state.semantic_chunking_lock:
            await run_in_threadpool(
                chunking.get_md_chunks,
                content_parsed,
                min_split_tokens=min_split_tokens,
                max_split_tokens=max_split_tokens,
                split_tokens_tolerance=split_tokens_tolerance,
                window_size=window_size,
            )
            
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(event="chunking_done", doc_id=doc_id, duration_ms=duration_ms,
                    num_chunks=chunking.chunk_size,
                    num_tokens=sum(chunking.token_count))
        
        return chunking.as_dict()
    
    except Exception as e:
        logger.error(event="chunking_error", error=str(e), doc_id=doc_id, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Error: {e}")

@app.post("/services/chunking/short_text")
async def process_(      
    request: Request,
    short_text: str = Form(...),
    doc_id: Optional[str] = Form(None)
):
    start_time = time.time()
    logger.info(event="chunking_start", endpoint="short_text", doc_id=doc_id)
    try:
        chunking = ChunkData()
        async with request.app.state.semantic_chunking_lock:
            await run_in_threadpool(chunking.get_txt_chunks, short_text)
            
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(event="chunking_done", doc_id=doc_id, duration_ms=duration_ms,
                    num_chunks=chunking.chunk_size,
                    num_tokens=sum(chunking.token_count))
        
        return chunking.as_dict()
    
    except Exception as e:
        logger.error(event="chunking_error", error=str(e), doc_id=doc_id, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Error: {e}")


@app.post("/services/chunking/doc_by_titles")
async def process_by_titles(
    content_parsed: str = Form(...),
    doc_id: Optional[str] = Form(None),
    max_split_tokens: int = Form(800, ge=1),
):
    start_time = time.time()
    logger.info(
        event="chunking_start",
        endpoint="doc_by_titles",
        doc_id=doc_id,
        max_split_tokens=max_split_tokens,
    )
    try:
        chunking = ChunkData()
        await run_in_threadpool(
            chunking.get_title_chunks,
            content_parsed,
            max_split_tokens=max_split_tokens,
        )

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            event="chunking_done",
            doc_id=doc_id,
            duration_ms=duration_ms,
            num_chunks=chunking.chunk_size,
            num_tokens=sum(chunking.token_count),
        )
        return chunking.as_dict()

    except Exception as e:
        logger.error(event="chunking_error", error=str(e), doc_id=doc_id, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Error: {e}")
