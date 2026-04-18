"""
EmotiScan: Ollama/Qwen3 LLM integration.
Supports streaming and non-streaming.
"""

import re
import httpx
from config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_NUM_CTX, OLLAMA_TEMPERATURE, GPU_LAYERS


class LLMService:
    """LLM service connecting to Ollama at localhost:8003."""

    def __init__(self, base_url: str = None, model: str = None, timeout: float = None):
        self.base_url = base_url or OLLAMA_URL
        self.model = model or OLLAMA_MODEL
        self.timeout = timeout or OLLAMA_TIMEOUT
        self._available = None

    @staticmethod
    def _clean_output(text: str) -> str:
        """Strip Qwen3 <think>...</think> reasoning blocks and tidy whitespace."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def _generate(self, prompt: str, stream: bool = False) -> str:
        """Call Ollama generate API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_gpu": GPU_LAYERS,
                "temperature": OLLAMA_TEMPERATURE,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        }
        try:
            if not stream:
                resp = httpx.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return self._clean_output(resp.json().get("response", ""))
            else:
                return self._stream_generate(payload)
        except Exception:
            return ""

    def _stream_generate(self, payload: dict) -> str:
        """Stream response from Ollama, return full text."""
        chunks = []
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        chunks.append(chunk)
                        if data.get("done", False):
                            break
        except Exception:
            pass
        return self._clean_output("".join(chunks))

    def stream_generate(self, prompt: str):
        """Yield chunks from Ollama streaming API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_gpu": GPU_LAYERS,
                "temperature": OLLAMA_TEMPERATURE,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        }
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()
                import json
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                        if data.get("done", False):
                            break
        except Exception:
            yield ""

    def research_emotion(self, emotion: str) -> str:
        """Generate research content about a detected emotion."""
        prompt = (
            f"You are an expert in psychology and neuroscience. "
            f"Provide a brief, fascinating research summary about the emotion '{emotion}'. "
            f"Include: what happens in the brain, evolutionary purpose, "
            f"and one surprising fact. Keep it under 200 words. Be engaging!"
        )
        result = self._generate(prompt)
        return result.strip() or ""

    def generate_mood_response(self, emotion: str, context: str = "") -> str:
        """Generate an empathetic/funny response to the detected emotion."""
        ctx = f" Context: {context}" if context else ""
        prompt = (
            f"The user is feeling '{emotion}'.{ctx} "
            f"Respond with empathy and a touch of humor. "
            f"Include a relevant joke or uplifting thought. Keep it short and warm."
        )
        result = self._generate(prompt)
        return result.strip() or ""

    def summarize_papers(self, papers: list, topic: str) -> str:
        """Summarize a list of research papers."""
        if not papers:
            return "No papers to summarize."

        paper_text = "\n".join(
            f"{i+1}. {p.get('title', 'Untitled')} - {p.get('abstract', '')[:200]}"
            for i, p in enumerate(papers[:5])
        )
        prompt = (
            f"Summarize these research papers about '{topic}' into a structured digest. "
            f"Include key findings and connections between papers.\n\n"
            f"Papers:\n{paper_text}\n\nWrite a concise summary:"
        )
        result = self._generate(prompt)
        return result.strip() or ""

    def summarize_digest(self, topic: str, papers: list, web_results: list) -> str:
        """Generate an LLM-powered research digest from ArXiv papers and web results."""
        papers_text = ""
        for i, p in enumerate(papers[:6], 1):
            title = p.get("title", "Untitled")
            authors = ", ".join(p.get("authors", [])[:2])
            abstract = p.get("abstract", "")[:300]
            score = round(p.get("relevance_score", 0), 2)
            papers_text += (
                f"{i}. {title}{f' ({authors})' if authors else ''}\n"
                f"   Abstract: {abstract}\n"
                f"   Relevance score: {score}\n\n"
            )

        web_text = ""
        for i, r in enumerate(web_results[:4], 1):
            title = r.get("title", "Untitled")
            snippet = r.get("snippet", r.get("abstract", ""))[:200]
            url = r.get("url", "")
            web_text += f"{i}. {title}\n   {snippet}\n   URL: {url}\n\n"

        if not papers_text and not web_text:
            return (
                f"## Research Digest: {topic}\n\n"
                "_No sources were found. Please try a different topic or enable more sources._"
            )

        sections = []
        if papers_text:
            sections.append(f"### ArXiv Papers\n{papers_text}")
        if web_text:
            sections.append(f"### Web Results\n{web_text}")
        sources_block = "\n".join(sections)

        prompt = (
            f"You are an expert research analyst. Based on the following sources about '{topic}', "
            f"write a comprehensive research digest in markdown format.\n"
            f"Your digest must include:\n"
            f"1. An **Overview** paragraph summarising the current state of research on this topic.\n"
            f"2. **Key Findings** — the most important results or claims from the papers/articles.\n"
            f"3. **Themes & Trends** — recurring ideas or directions across the sources.\n"
            f"4. **Takeaways** — practical implications or suggested future research directions.\n\n"
            f"Use markdown headings (##, ###) and bullet points. Be concise but informative.\n\n"
            f"{sources_block}\n"
            f"Now write the research digest:"
        )
        result = self._generate(prompt)
        return result.strip() or (
            f"## Research Digest: {topic}\n\n"
            "_LLM summarisation failed. Please check the Ollama connection._"
        )

    def chat(self, message: str, history: list = None) -> str:
        """General chat with context from history."""
        history = history or []
        history_text = ""
        for msg in history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"

        prompt = (
            f"You are EmotiScan's AI assistant, knowledgeable about emotions, "
            f"psychology, and research. Be helpful, fun, and a bit chaotic.\n\n"
            f"{history_text}user: {message}\nassistant:"
        )
        result = self._generate(prompt)
        return result.strip() or ""

    def compassionate_chat(self, message: str, history: list = None, session_context: dict = None) -> str:
        """Compassionate counselor chat with session context."""
        history = history or []
        session_context = session_context or {}

        history_text = ""
        for msg in history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"

        ctx_parts = []
        if session_context.get("transcript"):
            ctx_parts.append(f"What user said: {session_context['transcript'][:300]}")
        if session_context.get("summary"):
            ctx_parts.append(f"Session summary: {session_context['summary']}")
        if session_context.get("dominant_emotion"):
            ctx_parts.append(f"Detected emotion (face+voice+audio): {session_context['dominant_emotion']}")
        if session_context.get("compassionate_response"):
            ctx_parts.append(f"Earlier support given: {session_context['compassionate_response'][:200]}")
        context_str = " | ".join(ctx_parts) if ctx_parts else "No session context available."

        prompt = (
            "You are EmotiScan's supportive AI counselor. Your role is to help users process "
            "their emotions with warmth, empathy, and evidence-based suggestions. When responding: "
            "1) Acknowledge and validate their feelings "
            "2) Provide specific, actionable coping methods "
            "3) Reference relevant psychological research or techniques (CBT, mindfulness, etc.) "
            "4) Suggest concrete steps they can take right now "
            "5) Be warm and encouraging, not clinical.\n\n"
            f"Context: {context_str}\n\n"
            f"{history_text}user: {message}\nassistant:"
        )
        result = self._generate(prompt)
        return result.strip() if result else ""

    def suggest_solutions(self, emotion: str, problem_text: str = "",
                           face_emotion: str = "", voice_emotion: str = "") -> str:
        """Generate specific solution suggestions based on the user's actual
        problems (transcript) AND emotions from all modalities."""
        emo_detail = []
        if face_emotion and face_emotion != "neutral":
            emo_detail.append(f"face camera detected '{face_emotion}'")
        if voice_emotion and voice_emotion != "neutral":
            emo_detail.append(f"voice tone detected '{voice_emotion}'")
        if emotion and emotion != "neutral":
            emo_detail.append(f"overall emotion is '{emotion}'")
        emo_str = "; ".join(emo_detail) if emo_detail else f"emotion is '{emotion}'"

        prompt = (
            f"The user's {emo_str}.\n"
            f"{'What they talked about: ' + problem_text[:600] if problem_text else ''}\n\n"
            f"Provide 3 specific, actionable solutions. CRITICAL RULES:\n"
            f"1. Solutions MUST address the user's ACTUAL real-world problems as described "
            f"in their transcript — e.g. if they talked about a technical bug, suggest "
            f"debugging strategies; if they discussed a relationship issue, suggest communication "
            f"approaches; if they mentioned work stress, suggest workload management.\n"
            f"2. Do NOT just suggest generic therapy techniques (CBT, DBT, mindfulness) "
            f"unless the user specifically asked for emotional coping. The user wants "
            f"practical solutions to what they actually discussed.\n"
            f"3. You may include ONE brief emotional coping tip as a secondary suggestion, "
            f"but the primary solutions must be practical problem-solving for their situation.\n\n"
            f"Format as a numbered markdown list. Each solution: technique/approach name, "
            f"how to do it step-by-step, and why it addresses their specific problem."
        )
        result = self._generate(prompt)
        return result.strip() if result else ""
