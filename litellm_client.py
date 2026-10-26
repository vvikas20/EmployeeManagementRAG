import os
import yaml
import litellm
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

class LiteLLMClient:
    def __init__(self, config_path: str = "config.yaml"):
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            self.full_config = yaml.safe_load(f) or {}

        local_config_path = config_path.with_name("config.local.yaml")
        if local_config_path.is_file():
            with local_config_path.open("r", encoding="utf-8") as f:
                local_config = yaml.safe_load(f) or {}
            self.full_config = self._deep_merge(self.full_config, local_config)

        llm_config = self.full_config.get("llm", {})
        profiles = llm_config.get("profiles", {})
        if not profiles:
            profiles = {"ollama": self._legacy_ollama_profile(self.full_config)}

        self.active_profile = os.getenv("LLM_PROFILE", llm_config.get("active_profile", "ollama"))
        if self.active_profile not in profiles:
            raise ValueError(f"Unknown LLM profile '{self.active_profile}'.")

        self.profile_settings = profiles[self.active_profile]
        self.model_settings = {
            alias: dict(self.profile_settings.get(alias, {}))
            for alias in ("default-llm", "fast-llm")
        }

        # Support overriding Ollama base URL via environment variable (e.g. inside Docker)
        ollama_env_base = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_API_BASE"))
        if ollama_env_base and self.active_profile == "ollama":
            for alias in ("default-llm", "fast-llm"):
                if alias in self.model_settings:
                    self.model_settings[alias]["api_base"] = ollama_env_base
            self.profile_settings["api_base"] = ollama_env_base

        self.default_model = self.model_settings["default-llm"].get("model", "ollama/llama3.1")
        self.fast_model = self.model_settings["fast-llm"].get("model", "ollama/qwen2.5:1.5b")
        self.api_base = self.model_settings["default-llm"].get("api_base", "http://localhost:11434")

        if self.active_profile in ("hosted", "groq"):
            env_key = os.getenv("GROQ_API_KEY") if self.active_profile == "groq" else os.getenv("GEMINI_API_KEY")
            missing_keys = [
                alias for alias, settings in self.model_settings.items()
                if not (settings.get("api_key") or self.profile_settings.get("api_key") or env_key)
            ]
            if missing_keys:
                raise ValueError(
                    f"The {self.active_profile} LLM profile requires an API key in config.local.yaml "
                    f"(or {self.active_profile.upper()}_API_KEY environment variable) for: {', '.join(missing_keys)}."
                )

        # Configure LiteLLM global settings
        litellm.telemetry = False
        litellm.drop_params = True

        if os.getenv("DISABLE_SSL_VERIFY", "true").lower() in ("1", "true", "yes"):
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            litellm.ssl_verify = False

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = LiteLLMClient._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _legacy_ollama_profile(config: Dict[str, Any]) -> Dict[str, Any]:
        profile = {}
        for model in config.get("model_list", []):
            alias = model.get("model_name")
            if alias in ("default-llm", "fast-llm"):
                profile[alias] = dict(model.get("litellm_params", {}))
        return profile

    def _request_options(self, alias: str) -> Dict[str, Any]:
        if hasattr(self, "model_settings"):
            options = dict(self.model_settings.get(alias, {}))
            if hasattr(self, "profile_settings"):
                for key in ("api_key", "api_base"):
                    if not options.get(key) and self.profile_settings.get(key):
                        options[key] = self.profile_settings[key]
            if os.getenv("DISABLE_SSL_VERIFY", "true").lower() in ("1", "true", "yes"):
                options["ssl_verify"] = False
            return options

        model = self.fast_model if alias == "fast-llm" else self.default_model
        res = {"model": model, "api_base": getattr(self, "api_base", None)}
        if os.getenv("DISABLE_SSL_VERIFY", "true").lower() in ("1", "true", "yes"):
            res["ssl_verify"] = False
        return res

    def _safe_error(self, error: Exception) -> str:
        message = str(error)
        for options in getattr(self, "model_settings", {}).values():
            key = options.get("api_key")
            if key:
                message = message.replace(key, "[REDACTED]")
        profile_key = getattr(self, "profile_settings", {}).get("api_key")
        if profile_key:
            message = message.replace(profile_key, "[REDACTED]")
        return message

    def generate_completion(
        self, 
        prompt: str, 
        system_prompt: str = "You are an intelligent corporate HR and Employee Assistant AI.", 
        use_fast_model: bool = False,
        temperature: Optional[float] = None
    ) -> str:
        """Call LLM via LiteLLM abstraction."""
        alias = "fast-llm" if use_fast_model else "default-llm"
        options = self._request_options(alias)
        target_model = options.pop("model", self.fast_model if use_fast_model else self.default_model)
        if temperature is not None:
            options["temperature"] = temperature
        options.setdefault("temperature", 0.2)
        options.setdefault("max_tokens", 2048)
        
        try:
            response = litellm.completion(
                model=target_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                **options
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LiteLLM completion error on {target_model}: {self._safe_error(e)}")
            # Fallback to fast model or direct response
            try:
                fallback_options = self._request_options("fast-llm")
                fallback_model = fallback_options.pop("model", self.fast_model)
                fallback_options["temperature"] = 0.1
                fallback_options.setdefault("max_tokens", 2048)
                response = litellm.completion(
                    model=fallback_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    **fallback_options
                )
                return response.choices[0].message.content.strip()
            except Exception as e2:
                print(f"LiteLLM fallback failed on {self.fast_model}: {self._safe_error(e2)}")
                return f"LLM generation failed. Primary model: {target_model}. Error: {self._safe_error(e)}"

    def generate_completion_stream(
        self, 
        prompt: str, 
        system_prompt: str = "You are an intelligent corporate HR and Employee Assistant AI.", 
        use_fast_model: bool = False,
        temperature: Optional[float] = None
    ):
        """Yield text chunks from the LLM via LiteLLM."""
        alias = "fast-llm" if use_fast_model else "default-llm"
        options = self._request_options(alias)
        target_model = options.pop("model", self.fast_model if use_fast_model else self.default_model)
        if temperature is not None:
            options["temperature"] = temperature
        options.setdefault("temperature", 0.2)
        options.setdefault("max_tokens", 2048)
        options["stream"] = True
        
        try:
            response = litellm.completion(
                model=target_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                **options
            )
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            safe_error = self._safe_error(e)
            print(f"LiteLLM streaming error on {target_model}: {safe_error}")
            yield f"\n[Error streaming from LLM: {safe_error}]"

    def _fix_json_newlines(self, text: str) -> str:
        """Fix raw newlines inside JSON string values so json.loads() won't choke."""
        # Replace actual newline/tab characters inside JSON string values
        result = []
        in_string = False
        escape_next = False
        for char in text:
            if escape_next:
                result.append(char)
                escape_next = False
            elif char == '\\':
                result.append(char)
                escape_next = True
            elif char == '"' and not escape_next:
                in_string = not in_string
                result.append(char)
            elif in_string and char == '\n':
                result.append('\\n')
            elif in_string and char == '\r':
                result.append('\\r')
            elif in_string and char == '\t':
                result.append('\\t')
            else:
                result.append(char)
        return ''.join(result)

    def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> Any:
        """
        Call default-llm with OpenAI-style tool schemas.
        Returns the raw Message object so the caller can inspect .tool_calls and .content.
        On failure returns {"error": <message>}.
        Uses the active profile's default-llm, which must support tool calling.
        """
        options = self._request_options("default-llm")
        target_model = options.pop("model", self.default_model)
        if temperature is not None:
            options["temperature"] = temperature
        options.setdefault("temperature", 0.2)
        options.setdefault("max_tokens", 2048)
        try:
            response = litellm.completion(
                model=target_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                **options
            )
            return response.choices[0].message
        except Exception as e:
            safe_error = self._safe_error(e)
            print(f"LiteLLM tool-calling error on {target_model}: {safe_error}")
            return {"error": safe_error}

    def generate_with_tools_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> Any:
        """
        Call default-llm with OpenAI-style tool schemas and stream=True.
        Returns the generator, or {"error": <msg>} on failure.
        """
        options = self._request_options("default-llm")
        target_model = options.pop("model", self.default_model)
        if temperature is not None:
            options["temperature"] = temperature
        options.setdefault("temperature", 0.2)
        options.setdefault("max_tokens", 2048)
        options["stream"] = True
        try:
            response = litellm.completion(
                model=target_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                **options
            )
            return response
        except Exception as e:
            safe_error = self._safe_error(e)
            print(f"LiteLLM streaming tool-calling error: {safe_error}")
            return {"error": safe_error}

    def generate_json_structured(
        self, 
        prompt: str, 
        system_prompt: str, 
        use_fast_model: bool = True
    ) -> Dict[str, Any]:
        """Generate structured JSON response from LiteLLM."""
        json_system = system_prompt + "\nIMPORTANT: Return ONLY valid JSON output without markdown backticks or code fences."
        raw_text = self.generate_completion(prompt, json_system, use_fast_model=use_fast_model, temperature=0.1)
        
        # Clean markdown code blocks if present
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        # Extract JSON object boundaries
        start = clean_text.find("{")
        end = clean_text.rfind("}")
        if start != -1 and end != -1:
            clean_text = clean_text[start:end+1]
        
        # Attempt 1: direct parse
        try:
            return json.loads(clean_text)
        except Exception:
            pass
        
        # Attempt 2: fix raw newlines inside string values
        try:
            fixed = self._fix_json_newlines(clean_text)
            return json.loads(fixed)
        except Exception:
            pass
        
        # Attempt 3: regex strip control chars and retry
        try:
            sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', clean_text)
            return json.loads(sanitized)
        except Exception:
            pass

        return {"error": "Failed to parse JSON", "raw_output": raw_text}
