"""
Harmony chat with tools (Llama-Server Backend Edition)
Modified to support HTTP backend via llama.cpp server

/Users/ningqing/Library/Caches/llama.cpp/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf

llama-server -hf ggml-org/gpt-oss-20b-GGUF --n-cpu-moe 12 -c 32768 --jinja --no-mmap

"""

import atexit
import argparse
import asyncio
import datetime
import os
import json
import requests
import sys
from pathlib import Path

# --- 原始依赖保持不变 ---
try:
    import gnureadline as readline
except ImportError:
    import readline

import torch
import termcolor


from gpt_oss.tools import apply_patch
from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import ExaBackend
from gpt_oss.tools.python_docker.docker_tool import PythonTool


import requests
import json
import sys
from pathlib import Path


# ==========================================
# 🚑 紧急热修复：PythonTool 参数 Bug
# ==========================================
try:
    from gpt_oss.tools.python_docker.docker_tool import PythonTool

    # 1. 保存原始的“坏”方法
    _original_make_response = PythonTool._make_response

    # 2. 定义一个新的“好”方法，它能吃掉多余的 channel 参数
    def _patched_make_response(self, output, channel=None):
        try:
            # 先尝试正常传参（万一以后官方修好了）
            return _original_make_response(self, output, channel=channel)
        except TypeError:
            # 如果报错说不支持 channel，那我们就只传 output
            return _original_make_response(self, output)

    # 3. 偷梁换柱：把类的方法替换掉
    PythonTool._make_response = _patched_make_response
    print("✅ 已应用 PythonTool 热修复补丁")

except ImportError:
    pass



os.environ["EXA_API_KEY"] = "4bf3690f-ce7e-4966-aeee-67c2f938b0c6"

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)

REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

# --- 新增：LlamaServerGenerator ---
class LlamaServerGenerator:
    """
    Adapter that mimics the TokenGenerator interface but talks to a llama-server via HTTP.
    """
    def __init__(self, base_url, encoding):
        self.url = base_url
        self.encoding = encoding

    def generate(self, tokens, stop_tokens_list):
        """
        Yields predicted tokens given the input context tokens.
        """
        # 1. Decode context tokens back to string prompt
        prompt_str = self.encoding.decode(tokens)
        
        # 2. Construct Payload
        # 注意：stop 设为空列表，让模型把 <|end|> 等标签原样吐出来，
        # 不要让 Server 自作主张吞掉它们，否则 Parser 无法识别结束。
        payload = {
            "prompt": prompt_str,
            "n_predict": 4096,     
            "temperature": 0.7,   
            "stream": True,
            "stop": [], # ✅ 关键修正1：留空，让 Token 原样流出
            "cache_prompt": True 
        }

        # 3. HTTP Request
        try:
            # 这里的 self.url 应该是 http://127.0.0.1:8080/completion
            response = requests.post(self.url, json=payload, stream=True)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"\n[Error connecting to llama-server: {e}]", flush=True)
            return

        # 4. Stream processing
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    json_str = decoded_line[6:] # Remove "data: "
                    try:
                        data = json.loads(json_str)
                        
                        # Handle content
                        if "content" in data:
                            text_chunk = data["content"]
                            
                            # ✅ 关键修正2：加上 allowed_special="all"
                            # 这样 tiktoken 就能正确编码 <|channel|> 等特殊标签，不会报错 KeyError
                            chunk_tokens = self.encoding.encode(text_chunk, allowed_special="all")
                            
                            for t in chunk_tokens:
                                yield t

                        # Handle explicit stops from server
                        if data.get("stop"):
                            break
                            
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        print(f"[Error processing chunk: {e}]")

        # 5. ✅ 关键修正3：强制补发结束符 (The Kick)
        # 当 HTTP 流结束时，无论模型有没有吐出 <|end|>，我们都手动补发一个。
        # 这就像给 Parser 踢了一脚，告诉它：“别等了，这句话肯定说完了，快去执行工具调用！”
        end_token = self.encoding.encode("<|end|>", allowed_special="all")
        for t in end_token:
            yield t


def get_user_input():
    # 简化：假设在 Client 模式下不跑分布式
    return input()

def main(args):
    # 加载 Harmony 编码器 (用于处理 Prompt 格式和 Token 转换)
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # --- Backend Selection ---
    match args.backend:
        case "llama-server":
            # 这里实例化我们的新生成器
            generator = LlamaServerGenerator(args.server_url, encoding)
            print(termcolor.colored(f"Backend: Llama Server at {args.server_url}", "green"))
        case "triton":
            from gpt_oss.triton.model import TokenGenerator as TritonGenerator
            from gpt_oss.torch.utils import init_distributed
            device = init_distributed()
            generator = TritonGenerator(args.checkpoint, args.context, device)
        case "torch":
            from gpt_oss.torch.model import TokenGenerator as TorchGenerator
            from gpt_oss.torch.utils import init_distributed
            device = init_distributed()
            generator = TorchGenerator(args.checkpoint, device)
        case "vllm":
            from gpt_oss.vllm.token_generator import TokenGenerator as VLLMGenerator
            generator = VLLMGenerator(args.checkpoint, tensor_parallel_size=2)
        case _:
            raise ValueError(f"Invalid backend: {args.backend}")

    # --- System Message Construction (Unchanged) ---
    system_message_content = (
        SystemContent.new()
        .with_reasoning_effort(REASONING_EFFORT[args.reasoning_effort])
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    )

    # --- Tools Setup (Unchanged) ---
    if args.browser:
        backend = ExaBackend(source="web")
        browser_tool = SimpleBrowserTool(backend=backend)
        system_message_content = system_message_content.with_tools(browser_tool.tool_config)

    if args.python:
        python_tool = PythonTool()
        system_message_content = system_message_content.with_tools(python_tool.tool_config)

    system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)
    messages = [system_message]

    # --- Developer Message / Patch Tool (Unchanged) ---
    if args.apply_patch:
        apply_patch_instructions = Path(apply_patch.__file__).parent / "apply_patch.md"
        developer_message = ""
        if args.developer_message:
            developer_message = args.developer_message + "\n"
        developer_message += apply_patch_instructions.read_text()
        developer_message_content = (
            DeveloperContent.new()
            .with_instructions(developer_message)
            .with_function_tools([
                ToolDescription.new(
                    "apply_patch",
                    "Patch a file",
                    parameters={
                        "type": "string",
                        "description": "Formatted patch code",
                        "default": "*** Begin Patch\n*** End Patch\n",
                    }
                ),
            ])
        )
        messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message_content))
    elif args.developer_message:
        developer_message_content = DeveloperContent.new().with_instructions(args.developer_message)
        messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message_content))
    else:
        developer_message_content = None

    # --- Initial Print ---
    if args.raw:
        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation(conversation)
        system_message = encoding.decode(tokens)
        print(system_message, flush=True, end="")
        empty_user_message_tokens = encoding.render(Message.from_role_and_content(Role.USER, ""))
        user_message_start = encoding.decode(empty_user_message_tokens[:-1])
        user_message_end = encoding.decode(empty_user_message_tokens[-1:])
    else:
        print(termcolor.colored("System Message:", "cyan"), flush=True)
        print(termcolor.colored("Model Identity:", "cyan"), system_message_content.model_identity, flush=True)
        print(termcolor.colored("Backend:", "cyan"), args.backend, flush=True)
        if developer_message_content:
            print(termcolor.colored("Developer Message:", "yellow"), flush=True)
            print(developer_message_content.instructions, flush=True)

    MESSAGE_PADDING = 12

    # --- Main Chat Loop ---
    while True:
        last_message = messages[-1]
        
        # 1. Determine if we need user input
        if last_message.recipient is None:
            if args.raw:
                print(user_message_start, end="", flush=True)
                try:
                    user_message = get_user_input()
                except EOFError:
                    break
                print(user_message_end, flush=True, end="")
            else:
                print(termcolor.colored("User:".ljust(MESSAGE_PADDING), "red"), flush=True)
                try:
                    user_message = get_user_input()
                except EOFError:
                    print("\nExiting...")
                    break
            
            user_message = Message.from_role_and_content(Role.USER, user_message)
            messages.append(user_message)
        else:
            # 2. Handle Tool Execution (Logic Unchanged)
            # This is where the script shines: local tool execution driven by remote LLM
            if last_message.recipient.startswith("browser."):
                assert args.browser, "Browser tool is not enabled"
                tool_name = "Search"
                async def run_tool():
                    results = []
                    async for msg in browser_tool.process(last_message):
                        results.append(msg)
                    return results
                result = asyncio.run(run_tool())
                messages += result
                
            elif last_message.recipient.startswith("python"):
                assert args.python, "Python tool is not enabled"
                tool_name = "Python"
                async def run_tool():
                    results = []
                    async for msg in python_tool.process(last_message):
                        results.append(msg)
                    return results
                result = asyncio.run(run_tool())
                messages += result
                
            elif last_message.recipient == "functions.apply_patch":
                # ... (apply_patch logic unchanged) ...
                assert args.apply_patch, "Apply patch tool is not enabled"
                tool_name = "Apply Patch"
                text = last_message.content[0].text
                tool_output = None
                if text.startswith("{"):
                    import json
                    try:
                        some_dict = json.loads(text)
                        _, text = some_dict.popitem()
                    except Exception as e:
                        tool_output = f"Error parsing JSON: {e}"
                if tool_output is None:
                    try:
                        tool_output = apply_patch.apply_patch(text)
                    except Exception as e:
                        tool_output = f"Error applying patch: {e}"
                message = (
                    Message(
                        author=Author.new(Role.TOOL, last_message.recipient),
                        content=[TextContent(text=tool_output)]
                    ).with_recipient("assistant")
                )
                if last_message.channel:
                    message = message.with_channel(last_message.channel)
                result = [message]
                messages += result
            else:
                raise ValueError(f"Unknown tool or function call: {last_message.recipient}")

            # Print tool result
            if args.raw:
                rendered_result = encoding.render_conversation(Conversation.from_messages(result))
                print(encoding.decode(rendered_result), flush=True, end="")
            else:
                print(termcolor.colored(f"{tool_name} output:".ljust(MESSAGE_PADDING), "magenta"), flush=True)
                if tool_name == "Search" and not args.show_browser_results:
                    print("[Search results fed to the model]")
                else:
                    print(result[0].content[0].text)

        # 3. Prepare Prompt for Generation
        conversation = Conversation.from_messages(messages)
        # Render entire conversation to tokens using Harmony encoder
        tokens = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )

        if args.raw:
            print(encoding.decode(tokens[-2:]), flush=True, end="")

        # 4. Generate & Parse Response
        # The generator (LlamaServerGenerator) yields tokens
        parser = StreamableParser(encoding, role=Role.ASSISTANT)
        field_created = False
        current_output_text = ""
        output_text_delta_buffer = ""
        
        # Generator now calls the HTTP endpoint
        for predicted_token in generator.generate(tokens, encoding.stop_tokens_for_assistant_actions()):
            parser.process(predicted_token)
            
            if args.raw:
                print(encoding.decode([predicted_token]), end="", flush=True)
                continue

            # Standard UI logic (unchanged)
            if parser.state == StreamState.EXPECT_START:
                print("") 
                field_created = False

            if not parser.last_content_delta:
                continue

            if not field_created:
                field_created = True
                if parser.current_channel == "final":
                    print(termcolor.colored("Assistant:", "green"), flush=True)
                elif parser.current_recipient is not None:
                    print(termcolor.colored(f"Tool call to {parser.current_recipient}:", "cyan"), flush=True)
                else:
                    print(termcolor.colored("CoT:", "yellow"), flush=True)

            should_send_output_text_delta = True
            output_text_delta_buffer += parser.last_content_delta
            
            # Browser citation handling
            if args.browser:
                updated_output_text, _annotations, has_partial_citations = browser_tool.normalize_citations(current_output_text + output_text_delta_buffer)
                output_text_delta_buffer = updated_output_text[len(current_output_text):]
                if has_partial_citations:
                    should_send_output_text_delta = False
            
            if should_send_output_text_delta:
                print(output_text_delta_buffer, end="", flush=True)
                current_output_text += output_text_delta_buffer
                output_text_delta_buffer = ""

        # Update messages with the parsed assistant response
        messages += parser.messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chat example (Llama-Server Support)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Changed: checkpoint is only needed for local inference backends
    parser.add_argument(
        "checkpoint",
        metavar="FILE",
        type=str,
        nargs="?", # Made optional
        help="Path to the SafeTensors checkpoint (required for torch/vllm/triton)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8080/completion",
        help="URL for llama-server backend",
    )
    parser.add_argument(
        "-r", "--reasoning-effort", type=str, default="low", choices=["high", "medium", "low"],
    )
    parser.add_argument("-a", "--apply-patch", action="store_true")
    parser.add_argument("-b", "--browser", action="store_true")
    parser.add_argument("--show-browser-results", action="store_true")
    parser.add_argument("-p", "--python", action="store_true")
    parser.add_argument("--developer-message", default="")
    parser.add_argument("-c", "--context", type=int, default=8192)
    parser.add_argument("--raw", action="store_true")
    
    # Updated choices
    parser.add_argument(
        "--backend",
        type=str,
        default="llama-server",
        choices=["triton", "torch", "vllm", "llama-server"],
        help="Inference backend",
    )
    args = parser.parse_args()

    # check if checkpoint is needed
    if args.backend in ["triton", "torch", "vllm"] and not args.checkpoint:
        parser.error(f"--backend {args.backend} requires a checkpoint file.")

    if int(os.environ.get("WORLD_SIZE", 1)) == 1:
        histfile = os.path.join(os.path.expanduser("~"), ".chat_history")
        try:
            readline.read_history_file(histfile)
            readline.set_history_length(10000)
        except FileNotFoundError:
            pass
        atexit.register(readline.write_history_file, histfile)

    main(args)