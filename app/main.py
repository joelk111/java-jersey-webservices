"""
NLP Rules Engine - Main Application Entry Point

This module provides the Gradio web interface for the NLP-based
data quality rules creation system.

Usage:
    python -m app.main

Or:
    python app/main.py
"""

import os
import sys
import gradio as gr
from typing import Tuple, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import config
from app.conversation import ConversationManager
from app.field_matcher import FieldMatcher


class RulesEngineUI:
    """Gradio-based UI for the NLP Rules Engine."""

    def __init__(self):
        self.conversation = ConversationManager()
        self.field_matcher = FieldMatcher()

    def process_message(
        self,
        message: str,
        history: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], str, str, str]:
        """
        Process a user message and return updated state.

        Args:
            message: User's input message
            history: Chat history as list of (user, assistant) tuples

        Returns:
            Tuple of (updated_history, rules_csv, generated_code, field_matches)
        """
        if not message.strip():
            return history, "", "", ""

        try:
            # Process through conversation manager
            result = self.conversation.process_message(message)

            # Build response message
            response_parts = []

            # Main response text
            if result.get("message"):
                response_parts.append(result["message"])

            # Clarification request
            if result.get("clarification"):
                clarification = result["clarification"]
                question = clarification.get("question", "")
                options = clarification.get("options", [])
                if question:
                    response_parts.append(f"\n**Question:** {question}")
                if options:
                    response_parts.append("Options: " + ", ".join(options))

            # Field matches
            if result.get("matched_fields"):
                response_parts.append("\n**Matched Fields:**")
                for query, matches in result["matched_fields"].items():
                    if matches:
                        top_match = matches[0]
                        response_parts.append(f"- '{query}' -> {top_match[0]} (score: {top_match[1]})")

            # Generated rules
            if result.get("rules"):
                response_parts.append("\n**Generated Rule(s):**")
                for rule in result["rules"]:
                    response_parts.append(f"```json\n{rule}\n```")

            # Combine response
            response = "\n".join(response_parts) if response_parts else "I'm ready to help create data quality rules. Please describe the validation you need."

            # Update history
            history.append((message, response))

            # Get CSV and code outputs
            rules_csv = self.conversation.get_rules_csv()
            generated_code = "\n\n".join(self.conversation.state.generated_code)

            # Field matches summary
            field_matches = ""
            if result.get("matched_fields"):
                lines = []
                for query, matches in result["matched_fields"].items():
                    for match in matches[:3]:
                        lines.append(f"{query}: {match[0]} (score: {match[1]})")
                field_matches = "\n".join(lines)

            return history, rules_csv, generated_code, field_matches

        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            history.append((message, error_msg))
            return history, "", "", ""

    def clear_conversation(self) -> Tuple[List, str, str, str]:
        """Reset the conversation state."""
        self.conversation.reset()
        return [], "", "", ""

    def search_fields(self, query: str) -> str:
        """Search for fields matching a query."""
        if not query.strip():
            return ""

        results = self.field_matcher.match(query, limit=10)
        if not results:
            return "No matching fields found."

        lines = ["**Matching Fields:**"]
        for field, score, label in results:
            lines.append(f"- `{field}` (score: {score})")
            if label:
                lines.append(f"  Label: {label}")

        return "\n".join(lines)

    def list_tables(self) -> str:
        """List all available tables."""
        tables = self.field_matcher.list_tables()
        return "\n".join([f"- {t}" for t in tables])

    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface."""

        with gr.Blocks(
            title="NLP Rules Engine",
            theme=gr.themes.Soft()
        ) as interface:

            gr.Markdown("""
            # NLP Data Quality Rules Engine

            Create data quality rules using natural language. Describe the validation
            you need, and the system will generate the corresponding rule in CSV format.

            **Examples:**
            - "Create a rule for web_url that checks if it contains exclamation points and fail the test"
            - "Validate that email addresses have proper format"
            - "Ensure BVD ID numbers are not empty and start with a 2-letter country code"
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=400,
                        show_copy_button=True
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Your message",
                            placeholder="Describe the rule you want to create...",
                            lines=2,
                            scale=4
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)

                    with gr.Row():
                        clear_btn = gr.Button("Clear Conversation")

                with gr.Column(scale=1):
                    gr.Markdown("### Field Search")
                    field_search = gr.Textbox(
                        label="Search for fields",
                        placeholder="e.g., email, url, bvd"
                    )
                    field_results = gr.Markdown(label="Search Results")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Generated Rules (CSV)")
                    rules_output = gr.Code(
                        label="Rules CSV",
                        language="csv",
                        lines=10
                    )
                    download_csv = gr.Button("Download CSV")

                with gr.Column():
                    gr.Markdown("### Generated Code")
                    code_output = gr.Code(
                        label="Python Functions",
                        language="python",
                        lines=10
                    )

            with gr.Accordion("Field Matches", open=False):
                field_matches_output = gr.Textbox(
                    label="Matched Fields",
                    lines=5
                )

            with gr.Accordion("Available Tables", open=False):
                tables_display = gr.Markdown(self.list_tables())

            # Event handlers
            submit_btn.click(
                fn=self.process_message,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, rules_output, code_output, field_matches_output]
            ).then(
                fn=lambda: "",
                outputs=[msg_input]
            )

            msg_input.submit(
                fn=self.process_message,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, rules_output, code_output, field_matches_output]
            ).then(
                fn=lambda: "",
                outputs=[msg_input]
            )

            clear_btn.click(
                fn=self.clear_conversation,
                outputs=[chatbot, rules_output, code_output, field_matches_output]
            )

            field_search.submit(
                fn=self.search_fields,
                inputs=[field_search],
                outputs=[field_results]
            )

        return interface


def main():
    """Run the application."""
    print(f"Starting NLP Rules Engine...")
    print(f"Ollama host: {config.ollama_host}")
    print(f"Model: {config.ollama_model}")
    print(f"Field dictionary: {config.field_dictionary_path}")

    ui = RulesEngineUI()
    interface = ui.build_interface()

    interface.launch(
        server_name=config.app_host,
        server_port=config.app_port,
        share=False,
        show_error=config.debug
    )


if __name__ == "__main__":
    main()
