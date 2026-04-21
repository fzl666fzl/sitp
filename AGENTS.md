
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.



## 如何协助我处理本仓库
先说明思路，再进行修改。
除非我明确要求，否则不要修改代码。
把这份代码当作不熟悉的学生毕业设计项目来对待。
讲解代码时，使用简单中文。
先讲整体架构和模块职责，再讲具体实现细节。
针对我询问的每个文件，说明以下内容：
文件用途
核心类 / 函数
输入与输出
调用方与被调用方（谁调用它，它调用谁）
接下来最值得优先阅读的最少文件
如果代码含义不明确，直接说明不确定之处，不要随意猜测。
优先简洁易懂，避免抽象专业术语。


回答时优先按以下结构输出：
1. 文件/目录作用
2. 核心类和函数
3. 输入与输出
4. 调用关系
5. 执行流程
6. 下一步最值得阅读的文件
7. 不确定或可能有歧义的地方


当我请求解释项目时：
先给整体架构图式说明，
再给模块职责划分，
再追踪主流程，
最后再展开具体函数和细节。