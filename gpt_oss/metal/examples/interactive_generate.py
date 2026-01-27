#!/usr/bin/env python
"""
交互式GPT-OSS Metal生成脚本
可以持续输入提示词进行对话
"""

import os
from gpt_oss.metal import Context, Model

def main():
    # 配置设置
    MODEL_PATH = "gpt-oss-20b/metal/model.bin"  # 模型路径
    MAX_TOKENS = 100                           # 每次最大生成token数
    CONTEXT_LENGTH = 2048                      # 上下文长度
    
    print("🚀 GPT-OSS Metal 交互式生成器")
    print("=" * 50)
    print(f"📁 模型路径: {MODEL_PATH}")
    print(f"🎯 每次最大token数: {MAX_TOKENS}")
    print(f"📏 上下文长度: {CONTEXT_LENGTH}")
    print("💡 输入 'quit' 或 'exit' 退出程序")
    print("=" * 50)
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        print("请确保已下载模型文件到正确位置")
        return
    
    try:
        # 加载模型
        print("🔄 正在加载模型...")
        model = Model(MODEL_PATH)
        print("✅ 模型加载成功\n")
        
        while True:
            # 获取用户输入
            prompt = input("👤 请输入提示词: ").strip()
            
            # 退出条件
            if prompt.lower() in ['quit', 'exit', '退出']:
                print("👋 再见!")
                break
            
            if not prompt:
                print("⚠️ 提示词不能为空，请重新输入")
                continue
            
            try:
                # 为每次对话创建新的上下文
                print("🔄 正在创建上下文...")
                context = Context(model, context_length=CONTEXT_LENGTH)
                
                # 添加提示词
                context.append(prompt)
                prompt_tokens = context.num_tokens
                
                print(f"🤖 GPT-OSS: ", end='', flush=True)
                
                # 生成回复
                tokenizer = model.tokenizer
                generated_tokens = 0
                
                while context.num_tokens - prompt_tokens < MAX_TOKENS:
                    try:
                        token = context.sample()
                        context.append(token)
                        decoded = str(tokenizer.decode(token), encoding="utf-8")
                        print(decoded, end='', flush=True)
                        generated_tokens += 1
                        
                        # 检查是否完成了一个句子
                        if decoded in ['.', '!', '?', '\n']:
                            # 可以选择在句子结束时停止
                            pass
                            
                    except Exception as e:
                        print(f"\n⚠️ 生成过程中出现错误: {e}")
                        break
                
                print(f"\n📊 [生成了 {generated_tokens} 个token]\n")
                
            except Exception as e:
                print(f"❌ 处理提示词时发生错误: {e}")
                print("💡 尝试使用更短的提示词或重启程序\n")
                
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("\n💡 可能的解决方案:")
        print("1. 检查是否有足够的内存 (推荐24GB+)")
        print("2. 尝试减少CONTEXT_LENGTH的值")
        print("3. 关闭其他应用程序释放内存")

if __name__ == '__main__':
    main()
