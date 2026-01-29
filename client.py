import base64
import os
from time import sleep
import requests
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class Client:

    def __init__(self):
        self.run_pod_api_key = os.getenv("RUN_POD_API_KEY")
        self.run_pod_id = os.getenv("RUN_POD_ID")


    def save_image(self, task_id: str) -> None:
        response = requests.get(
            url=f"https://api.runpod.ai/v2/{self.run_pod_id}/status/{task_id}",
            headers={
                "Authorization": f"Bearer {self.run_pod_api_key}",
                "Content-Type": "application/json"
            }
        )
        data = response.json()
        data_url = data["output"]["image"]
        _, b64data = data_url.split(",", 1)
        image_bytes = base64.b64decode(b64data)
        with open(f"./output/{task_id}.png", "wb") as f:
            f.write(image_bytes)

    def send_job(self, images: list[str], prompt: str) -> str:
        response = requests.post(
            url=f"https://api.runpod.ai/v2/{self.run_pod_id}/run",
            json={
                  "input": {
                    "images": images,
                    "prompt": prompt,
                  }
                },
            headers={
                "Authorization": f"Bearer {self.run_pod_api_key}",
                "Content-Type": "application/json"
            }
        )
        return response.json()["id"]

    def check_status(self, task_id: str) -> str:
        status = "IN_PROGRESS"
        while status not in ("FAILED", "COMPLETED"):
            sleep(5)
            response = requests.get(
                url=f"https://api.runpod.ai/v2/{self.run_pod_id}/status/{task_id}",
                headers={
                    "Authorization": f"Bearer {self.run_pod_api_key}",
                    "Content-Type": "application/json"
                }
            )
            data = response.json()
            status = data["status"]

            if status in ("FAILED", "COMPLETED"):
                if status == "FAILED":
                    logger.error(f"Job failed: {data['error']}")
                return status
            logger.info(f"Status: {status}, waiting for completion ...")
        return status


    def generate(self, images: list[str], prompt: str) -> str:
        task_id = self.send_job(images=images, prompt=prompt)
        status = self.check_status(task_id=task_id)
        match status:
            case "COMPLETED":
                print("Job completed successfully!")
                self.save_image(task_id=task_id)
                print(f"Image {task_id}.png saved successfully!")
        return status
