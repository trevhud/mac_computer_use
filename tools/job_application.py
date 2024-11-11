from typing import Any, Literal, TypedDict, Dict
from pathlib import Path
import base64
from openai import OpenAI
from openai.types.beta import Thread
from openai.types.beta.threads import Run
from fpdf import FPDF
import os
from datetime import datetime

from anthropic.types.beta import BetaToolUnionParam
from .base import BaseAnthropicTool, ToolError, ToolResult

class JobApplicationAction(TypedDict):
    action: Literal["save_jd", "generate_resume", "generate_cover_letter"]
    text: str | None

class JobApplicationTool(BaseAnthropicTool):
    """
    A tool that handles job applications by saving job descriptions and generating 
    customized resumes and cover letters using OpenAI.
    """
    
    name: Literal["job_application"] = "job_application"
    api_type: Literal["custom"] = "custom"
    
    def __init__(self):
        super().__init__()
        self.openai_client: OpenAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.assistant_id: str = "asst_LeF46BdHOuVGcSG8U900tv4C"
        self.stored_jd: str | None = None
        self.output_dir: Path = Path("/tmp/job_applications")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_params(self) -> BetaToolUnionParam:
        return {
            "name": self.name,
            "type": self.api_type,
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["save_jd", "generate_resume", "generate_cover_letter"],
                        "description": "The action to perform"
                    },
                    "text": {
                        "type": "string",
                        "description": "The text content (required for save_jd action)"
                    }
                },
                "required": ["action"]
            }
        }

    async def __call__(
        self,
        *,
        action: Literal["save_jd", "generate_resume", "generate_cover_letter"],
        text: str | None = None,
        **kwargs: Dict[str, Any],
    ) -> ToolResult:
        if action == "save_jd":
            if not text:
                raise ToolError("Job description text is required")
            self.stored_jd = text
            return ToolResult(output="Job description saved successfully")

        if action in ["generate_resume", "generate_cover_letter"]:
            if not self.stored_jd:
                raise ToolError("No job description stored. Please save a job description first.")
            
            # Create a thread and send the job description
            thread: Thread = await self.openai_client.beta.threads.create()
            await self.openai_client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"Job Description:\n{self.stored_jd}\n\nPlease generate a {'resume' if action == 'generate_resume' else 'cover letter'} for this position."
            )

            # Run the assistant
            run: Run = await self.openai_client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )

            # Wait for completion
            while run.status in ["queued", "in_progress"]:
                run = await self.openai_client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )

            if run.status != "completed":
                raise ToolError(f"Assistant run failed with status: {run.status}")

            # Get the response
            messages: list[Any] = await self.openai_client.beta.threads.messages.list(
                thread_id=thread.id
            )
            generated_text: str = messages.data[0].content[0].text.value

            # Generate PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Split text into lines and add to PDF
            lines: list[str] = generated_text.split('\n')
            for line in lines:
                pdf.multi_cell(w=0, h=10, txt=line)

            # Save PDF
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{action}_{timestamp}.pdf"
            pdf_path = self.output_dir / filename
            pdf.output(str(pdf_path))

            # Read PDF and convert to base64
            with open(pdf_path, "rb") as f:
                pdf_base64 = base64.b64encode(f.read()).decode()

            return ToolResult(
                output=f"Generated {action.replace('_', ' ')} as PDF",
                base64_image=pdf_base64
            )

        raise ToolError(f"Invalid action: {action}")