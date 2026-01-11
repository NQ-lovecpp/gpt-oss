#!/usr/bin/env python
"""
简化的GPT-OSS Metal生成脚本
可以直接运行，无需命令行参数
"""

import os
from gpt_oss.metal import Context, Model

def main():
    # 配置设置 - 可以根据需要修改这些值
    MODEL_PATH = "gpt-oss-20b/metal/model.bin"  # 模型路径
    PROMPT = "Hello, how are you?"              # 提示词
    MAX_TOKENS = 100                           # 最大生成token数
    CONTEXT_LENGTH = 2048                      # 上下文长度 (减少内存使用)
    
    print("🚀 启动 GPT-OSS Metal 生成器...")
    print(f"📁 模型路径: {MODEL_PATH}")
    print(f"💬 提示词: {PROMPT}")
    print(f"🎯 最大token数: {MAX_TOKENS}")
    print(f"📏 上下文长度: {CONTEXT_LENGTH}")
    print("-" * 50)
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        print("请确保已下载模型文件到正确位置")
        return
    
    try:
        # 加载模型
        print("🔄 正在加载模型...")
        model = Model(MODEL_PATH)
        print("✅ 模型加载成功")
        
        # 创建上下文
        print("🔄 正在创建上下文...")
        context = Context(model, context_length=CONTEXT_LENGTH)
        print("✅ 上下文创建成功")
        
        # 添加提示词
        context.append(PROMPT)
        print(f"📝 Token IDs: {context.tokens}")
        prompt_tokens = context.num_tokens
        
        print(f"\n💭 生成的文本:\n{PROMPT}", end='', flush=True)
        
        # 生成文本
        tokenizer = model.tokenizer
        generated_tokens = 0
        
        while context.num_tokens - prompt_tokens < MAX_TOKENS:
            try:
                token = context.sample()
                context.append(token)
                decoded = str(tokenizer.decode(token), encoding="utf-8")
                print(decoded, end='', flush=True)
                generated_tokens += 1
                
                # 可选：在每10个token后添加换行符以便观察
                if generated_tokens % 50 == 0:
                    print(f"\n[已生成 {generated_tokens} tokens]", end='', flush=True)
                    
            except Exception as e:
                print(f"\n⚠️ 生成过程中出现错误: {e}")
                break
        
        print(f"\n\n✅ 生成完成! 总共生成了 {generated_tokens} 个token")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print("\n💡 可能的解决方案:")
        print("1. 检查是否有足够的内存 (推荐24GB+)")
        print("2. 尝试减少CONTEXT_LENGTH的值")
        print("3. 关闭其他应用程序释放内存")

if __name__ == '__main__':
    main()
