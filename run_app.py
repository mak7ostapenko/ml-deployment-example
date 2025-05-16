import argparse

import uvicorn
from app.config import app_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app.")
    
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload."
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of worker processes."
    )
    args = parser.parse_args()
    
    uvicorn.run(
        "app.main:app", 
        host=app_cfg.hostname, 
        port=app_cfg.http_port, 
        reload=args.reload,
        workers=args.num_workers,
    )