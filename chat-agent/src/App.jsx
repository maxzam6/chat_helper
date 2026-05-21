import { useCallback, useEffect, useMemo, useState } from 'react'
import './App.css'

const EMPTY_CONTEXT = JSON.stringify(
  {
    recent_messages: [
      { role: 'target', content: '哦' },
      { role: 'me', content: '你是不是不想聊了' },
    ],
    previous_recent_messages: [],
  },
  null,
  2,
)

function App() {
  const [currentUserId, setCurrentUserId] = useState(
    () => localStorage.getItem('chat_helper_current_user') || 'A001',
  )
  const [userInput, setUserInput] = useState('她回我哦，我该怎么回？')
  const [chatContextText, setChatContextText] = useState(EMPTY_CONTEXT)
  const [screenshotBase64, setScreenshotBase64] = useState('')
  const [screenshotName, setScreenshotName] = useState('')
  const [result, setResult] = useState(null)
  const [memories, setMemories] = useState([])
  const [workingMemory, setWorkingMemory] = useState([])
  const [userSuggestions, setUserSuggestions] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const canSubmit = useMemo(() => userInput.trim() || screenshotBase64, [userInput, screenshotBase64])

  useEffect(() => {
    localStorage.setItem('chat_helper_current_user', currentUserId)
  }, [currentUserId])

  const fetchUserSuggestions = useCallback(async (query) => {
    try {
      const response = await fetch(`/api/users/suggest?query=${encodeURIComponent(query)}&limit=5`)
      if (!response.ok) return
      const data = await response.json()
      setUserSuggestions(data.suggestions || [])
    } catch {
      setUserSuggestions([])
    }
  }, [])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (!currentUserId.trim()) {
        setUserSuggestions([])
        return
      }
      fetchUserSuggestions(currentUserId)
    }, 300)
    return () => window.clearTimeout(timer)
  }, [currentUserId, fetchUserSuggestions])

  const refreshMemoryPanels = async (userId) => {
    if (!userId) return
    try {
      const [memoryResponse, workingResponse] = await Promise.all([
        fetch(`/api/users/${encodeURIComponent(userId)}/memories`),
        fetch(`/api/users/${encodeURIComponent(userId)}/working-memory`),
      ])
      if (memoryResponse.ok) {
        const memoryData = await memoryResponse.json()
        setMemories(memoryData.memories || [])
      }
      if (workingResponse.ok) {
        const workingData = await workingResponse.json()
        setWorkingMemory(workingData.working_memory || [])
      }
    } catch {
      // The main Agent call should still be usable if the side panels fail.
    }
  }

  const parseChatContext = () => {
    const text = chatContextText.trim()
    if (!text) return undefined
    const parsed = JSON.parse(text)
    if (Array.isArray(parsed)) {
      return { recent_messages: parsed, previous_recent_messages: [] }
    }
    return parsed
  }

  const handleScreenshotFile = (file) => {
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      setScreenshotBase64(String(reader.result || ''))
      setScreenshotName(file.name)
    }
    reader.readAsDataURL(file)
  }

  const handlePaste = (event) => {
    const imageItem = Array.from(event.clipboardData?.items || []).find((item) =>
      item.type.startsWith('image/'),
    )
    if (imageItem) {
      const file = imageItem.getAsFile()
      handleScreenshotFile(file)
    }
  }

  const submitToAgent = async () => {
    if (!canSubmit) return
    setIsLoading(true)
    setError('')
    setResult(null)

    try {
      const chatContext = parseChatContext()
      const payload = {
        me_id: 'default',
        user_input: userInput,
      }
      if (currentUserId.trim()) payload.current_user_id = currentUserId.trim()
      if (chatContext) payload.chat_context = chatContext
      if (screenshotBase64) payload.screenshot_base64 = screenshotBase64

      const response = await fetch('/api/agent/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Agent request failed')

      setResult(data)
      setUserSuggestions(data.user_id_suggestions || [])
      await refreshMemoryPanels(data.active_user_id || currentUserId.trim())
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setIsLoading(false)
    }
  }

  const clearScreenshot = () => {
    setScreenshotBase64('')
    setScreenshotName('')
  }

  return (
    <main className="app-shell" onPaste={handlePaste}>
      <section className="toolbar">
        <div>
          <p className="eyebrow">Chat Helper Agent</p>
          <h1>长期记忆对话调度台</h1>
        </div>
        <div className="status-pill">{isLoading ? '运行中' : '待运行'}</div>
      </section>

      <section className="workspace">
        <div className="panel compose-panel">
          <div className="field">
            <label htmlFor="current-user">当前聊天对象 ID</label>
            <input
              id="current-user"
              value={currentUserId}
              onChange={(event) => setCurrentUserId(event.target.value)}
              placeholder="例如 A001；普通问答可留空"
            />
          </div>

          {userSuggestions.length > 0 && (
            <div className="suggestions">
              <span>可能是已有用户：</span>
              {userSuggestions.map((item) => (
                <button
                  type="button"
                  key={item.user_id}
                  onClick={() => setCurrentUserId(item.user_id)}
                >
                  {item.user_id} · {(item.score * 100).toFixed(0)}%
                </button>
              ))}
            </div>
          )}

          <div className="field">
            <label htmlFor="user-input">你的需求</label>
            <textarea
              id="user-input"
              rows={4}
              value={userInput}
              onChange={(event) => setUserInput(event.target.value)}
              placeholder="例如：她回我哦，我该怎么回？"
            />
          </div>

          <div className="field">
            <label htmlFor="chat-context">聊天上下文 JSON</label>
            <textarea
              id="chat-context"
              rows={9}
              value={chatContextText}
              onChange={(event) => setChatContextText(event.target.value)}
              spellCheck="false"
            />
          </div>

          <div className="upload-row">
            <label className="upload-button">
              上传截图
              <input
                type="file"
                accept="image/*"
                onChange={(event) => handleScreenshotFile(event.target.files?.[0])}
              />
            </label>
            <span className="upload-hint">
              {screenshotName
                ? `${screenshotName} 已交给后端 Vision LLM 处理`
                : '也可以直接粘贴截图；前端不做 OCR'}
            </span>
            {screenshotName && (
              <button type="button" className="ghost-button" onClick={clearScreenshot}>
                移除截图
              </button>
            )}
          </div>

          <div className="actions">
            <button type="button" className="primary-button" disabled={isLoading || !canSubmit} onClick={submitToAgent}>
              {isLoading ? '处理中...' : '运行 Agent'}
            </button>
            <button type="button" className="ghost-button" onClick={() => setChatContextText(EMPTY_CONTEXT)}>
              填入示例上下文
            </button>
          </div>

          {error && <div className="error-box">{error}</div>}
        </div>

        <div className="panel result-panel">
          <h2>本轮输出</h2>
          {!result && <p className="empty-text">运行后会显示回复、意图、记忆写入和检索调试信息。</p>}
          {result && (
            <div className="result-stack">
              <div className="reply-box">
                <span>建议回复</span>
                <p>{result.reply?.content || '本轮没有生成回复内容'}</p>
              </div>
              <dl className="metrics">
                <div>
                  <dt>状态</dt>
                  <dd>{result.status || '-'}</dd>
                </div>
                <div>
                  <dt>意图</dt>
                  <dd>{result.intent || '-'}</dd>
                </div>
                <div>
                  <dt>任务列表</dt>
                  <dd>{(result.task_list || []).join(' / ') || '-'}</dd>
                </div>
                <div>
                  <dt>召回 query</dt>
                  <dd>{result.retrieval_query || '-'}</dd>
                </div>
                <div>
                  <dt>新增记忆 ID</dt>
                  <dd>{(result.saved_memory_ids || []).join(', ') || '-'}</dd>
                </div>
              </dl>
              <details>
                <summary>调试 JSON</summary>
                <pre>{JSON.stringify(result, null, 2)}</pre>
              </details>
            </div>
          )}
        </div>
      </section>

      <section className="memory-grid">
        <div className="panel">
          <h2>长期记忆</h2>
          {memories.length === 0 && <p className="empty-text">当前用户暂无长期记忆。</p>}
          <div className="memory-list">
            {memories.map((memory) => (
              <article className="memory-item" key={memory.id}>
                <div>
                  <strong>#{memory.id} {memory.memory_type || 'unknown'}</strong>
                  <span>{memory.memory_status} · {Number(memory.confidence).toFixed(2)}</span>
                </div>
                <p>{memory.content}</p>
              </article>
            ))}
          </div>
        </div>

        <div className="panel">
          <h2>Working Memory</h2>
          {workingMemory.length === 0 && <p className="empty-text">当前没有短期 observation。</p>}
          <div className="memory-list">
            {workingMemory.map((item) => (
              <article className="memory-item" key={item.id}>
                <div>
                  <strong>#{item.id}</strong>
                  <span>age {item.age} / ttl {item.ttl}</span>
                </div>
                <p>{item.content}</p>
              </article>
            ))}
          </div>
        </div>
      </section>
    </main>
  )
}

export default App
