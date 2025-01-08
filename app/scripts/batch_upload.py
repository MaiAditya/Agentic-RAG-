import asyncio
import aiohttp
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFBatchUploader:
    def __init__(self, api_url: str = "http://localhost:8000", max_concurrent: int = 5):
        self.api_url = api_url
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.progress_bar = None

    async def upload_single_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Upload a single PDF file to the API."""
        async with self.semaphore:
            try:
                async with aiohttp.ClientSession() as session:
                    data = aiohttp.FormData()
                    data.add_field('file',
                                 open(file_path, 'rb'),
                                 filename=file_path.name,
                                 content_type='application/pdf')

                    async with session.post(f"{self.api_url}/process-pdf", data=data) as response:
                        result = await response.json()
                        
                        if response.status == 200:
                            logger.info(f"Successfully processed {file_path.name}")
                            self.progress_bar.update(1)
                            return {"status": "success", "file": file_path.name, "result": result}
                        else:
                            logger.error(f"Failed to process {file_path.name}: {result}")
                            return {"status": "error", "file": file_path.name, "error": result}

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
                return {"status": "error", "file": file_path.name, "error": str(e)}

    async def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Process all PDF files in the given directory concurrently."""
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")

        # Get all PDF files
        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")
        self.progress_bar = tqdm(total=len(pdf_files), desc="Processing PDFs")

        # Create tasks for all PDF files
        tasks = [self.upload_single_pdf(pdf_file) for pdf_file in pdf_files]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        self.progress_bar.close()
        return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch upload PDFs to processing API")
    parser.add_argument("directory", help="Directory containing PDF files")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent uploads")
    args = parser.parse_args()

    uploader = PDFBatchUploader(api_url=args.api_url, max_concurrent=args.max_concurrent)
    
    try:
        # Run the async process
        results = asyncio.run(uploader.process_directory(args.directory))
        
        # Print summary
        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = sum(1 for r in results if r.get("status") == "error")
        
        print("\nProcessing Summary:")
        print(f"Total files: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {error_count}")
        
        # Print errors if any
        if error_count > 0:
            print("\nFailed files:")
            for result in results:
                if result.get("status") == "error":
                    print(f"- {result['file']}: {result['error']}")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 