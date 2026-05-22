import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'

const DEFAULT_USER = ''
const FIXED_SCREENSHOT_REGION = {
  left: 420,
  top: 80,
  width: 1000,
  height: 820,
}

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [currentUser, setCurrentUser] = useState(
    () => localStorage.getItem('chat_current_user') || DEFAULT_USER,
  )
  const [showUserDropdown, setShowUserDropdown] = useState(false)
  const [newUserName, setNewUserName] = useState('')
  const [userSuggestions, setUserSuggestions] = useState([])
  const [screenshotBase64, setScreenshotBase64] = useState('')
  const [screenshotName, setScreenshotName] = useState('')
  const [lastResult, setLastResult] = useState(null)
  const [hotkeyStatus, setHotkeyStatus] = useState(null)
  const [prepared, setPrepared] = useState(false)
  const fileInputRef = useRef(null)
  const pollTimerRef = useRef(null)
  const lastResultGenerationRef = useRef(0)
  const hotkeyLabel = hotkeyStatus?.hotkey || 'Ctrl + Shift + Y'

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      window.clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }, [])

  const refreshCaptureStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/capture/status')
      if (!response.ok) return null
      const data = await response.json()
      setHotkeyStatus(data)
      return data
    } catch {
      return null
    }
  }, [])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      refreshCaptureStatus()
    }, 0)
    return () => {
      window.clearTimeout(timer)
      stopPolling()
    }
  }, [refreshCaptureStatus, stopPolling])

  useEffect(() => {
    if (currentUser) {
      localStorage.setItem('chat_current_user', currentUser)
    } else {
      localStorage.removeItem('chat_current_user')
    }
  }, [currentUser])

  const startPolling = () => {
    stopPolling()
    pollTimerRef.current = window.setInterval(async () => {
      const status = await refreshCaptureStatus()
      if (!status) return
      if (status.running) {
        setIsLoading(true)
        return
      }
      if (status.latest_error && status.latest_error !== 'cancel_requested') {
        setIsLoading(false)
        appendAssistantMessage(`请求失败：${status.latest_error}`, true)
        stopPolling()
        return
      }
      const resultGeneration = status.latest_result_generation || 0
      if (status.latest_result && resultGeneration > lastResultGenerationRef.current) {
        lastResultGenerationRef.current = resultGeneration
        setIsLoading(false)
        handleAgentResult(status.latest_result)
        stopPolling()
      }
    }, 900)
  }

  const suggestUsers = async (query) => {
    if (!query.trim()) {
      setUserSuggestions([])
      return []
    }
    try {
      const response = await fetch(`/api/users/suggest?query=${encodeURIComponent(query)}&limit=5`)
      if (!response.ok) return []
      const data = await response.json()
      const suggestions = data.suggestions || []
      setUserSuggestions(suggestions)
      return suggestions
    } catch {
      setUserSuggestions([])
      return []
    }
  }

  const buildPayload = (content) => {
    const payload = {
      me_id: 'default',
      user_input: content,
      screenshot_region: FIXED_SCREENSHOT_REGION,
    }
    if (currentUser.trim()) {
      payload.current_user_id = currentUser.trim()
    }
    if (screenshotBase64) {
      payload.screenshot_base64 = screenshotBase64
    }
    return payload
  }

  const prepareHotkeyCapture = async () => {
    const content = input.trim()
    const payload = buildPayload(content)
    const response = await fetch('/api/capture/context', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    const data = await response.json()
    if (!response.ok) {
      throw new Error(data.detail || 'Capture context update failed')
    }
    setHotkeyStatus(data)
    if (typeof data.latest_result_generation === 'number') {
      lastResultGenerationRef.current = Math.max(
        lastResultGenerationRef.current,
        data.latest_result_generation,
      )
    }
    setPrepared(true)
    startPolling()
  }

  const sendMessage = async () => {
    const content = input.trim()

    setMessages((prev) => [
      ...prev,
      {
        role: 'user',
        content: content || `[等待快捷键截图分析${screenshotName ? `：${screenshotName}` : ''}]`,
        user: currentUser || '未选择',
      },
    ])
    setIsLoading(false)

    try {
      await prepareHotkeyCapture()
      appendAssistantMessage(
        `已准备好。请切回微信/QQ窗口，按 ${hotkeyLabel} 开始分析。`,
      )
    } catch (error) {
      appendAssistantMessage(`请求失败：${error.message}`, true)
    }
  }

  const stopGeneration = async () => {
    stopPolling()
    setIsLoading(false)
    setPrepared(false)
    try {
      await fetch('/api/capture/cancel', { method: 'POST' })
    } catch {
      // Best-effort cancellation; an in-flight model request may finish later.
    }
    appendAssistantMessage('已停止等待本轮生成。')
  }

  const handleAgentResult = (data) => {
    setLastResult(data)
    if (data.user_id_suggestions?.length) {
      setUserSuggestions(data.user_id_suggestions)
    }
    if (data.status === 'missing_user_input') {
      const message = data.reply?.content || '请先输入你需要我帮什么。'
      window.alert(message)
      appendAssistantMessage(message, true)
      setPrepared(false)
      return
    }
    if (data.is_valid_chat_window === false) {
      window.alert('未检测到有效聊天窗口，请将聊天窗口放入识别区域后再试。')
    }
    if (data.user_id_change_detected && data.recognized_user_id) {
      const shouldSwitch = window.confirm(
        `检测到当前聊天对象可能是 ${data.recognized_user_id}。\n是否切换为这个新 ID？`,
      )
      if (shouldSwitch) {
        setCurrentUser(data.recognized_user_id)
        appendAssistantMessage(`已切换到 ${data.recognized_user_id}，请再次按快捷键分析。`)
        setPrepared(false)
        return
      }
    }

    appendAssistantMessage(data.reply?.content || '这轮没有生成可发送的回复。')
    setScreenshotBase64('')
    setScreenshotName('')
  }

  const appendAssistantMessage = (content, error = false) => {
    setMessages((prev) => [
      ...prev,
      {
        role: 'assistant',
        content,
        error,
      },
    ])
  }

  const confirmSwitch = async () => {
    const nextUser = newUserName.trim()
    if (!nextUser) return
    const suggestions = await suggestUsers(nextUser)
    if (suggestions.length > 0 && suggestions[0].user_id !== nextUser) {
      const useSuggested = window.confirm(
        `可能已有相似用户：${suggestions[0].user_id}\n是否切换到这个已有用户？`,
      )
      setCurrentUser(useSuggested ? suggestions[0].user_id : nextUser)
    } else {
      setCurrentUser(nextUser)
    }
    setMessages([])
    setNewUserName('')
    setShowUserDropdown(false)
    setPrepared(false)
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

  const onPaste = (event) => {
    const imageItem = Array.from(event.clipboardData?.items || []).find((item) =>
      item.type.startsWith('image/'),
    )
    if (!imageItem) return
    const file = imageItem.getAsFile()
    handleScreenshotFile(file)
  }

  return (
    <div className="page-shell" onPaste={onPaste}>
      <div className="capture-overlay" aria-hidden="true">
        请将微信/QQ聊天窗口放入此区域，按 {hotkeyLabel} 分析
      </div>
      <div className="chat-card">
        <header className="chat-header">
          <div className="header-icon">✦</div>
          <h1>高情商聊天助手</h1>
          <p>先准备，再切回聊天窗口按快捷键分析</p>
        </header>

        <section className="user-zone">
          <div className="user-chip">
            <div className="avatar">☺</div>
            <div>
              <div className="chip-label">正在分析</div>
              <div className="chip-value">{currentUser || '未选择用户'}</div>
            </div>
          </div>
          <button className="switch-button" onClick={() => setShowUserDropdown(true)}>
            切换用户
          </button>
        </section>

        {showUserDropdown && (
          <section className="switch-panel">
            <label htmlFor="new-user">输入新的用户 ID</label>
            <input
              id="new-user"
              value={newUserName}
              onChange={(event) => {
                setNewUserName(event.target.value)
                suggestUsers(event.target.value)
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter') confirmSwitch()
              }}
              autoFocus
              placeholder="例如 A001"
            />
            {userSuggestions.length > 0 && (
              <div className="suggestion-row">
                {userSuggestions.map((item) => (
                  <button key={item.user_id} onClick={() => setNewUserName(item.user_id)}>
                    {item.user_id} · {(item.score * 100).toFixed(0)}%
                  </button>
                ))}
              </div>
            )}
            <div className="switch-actions">
              <button className="soft-button" onClick={() => setShowUserDropdown(false)}>
                取消
              </button>
              <button className="solid-button" onClick={confirmSwitch}>
                确认切换
              </button>
            </div>
          </section>
        )}

        <section className="message-list">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">♡</div>
              <div>还没有消息</div>
              <span>输入需求，准备后切回聊天窗口按快捷键</span>
            </div>
          )}
          {messages.map((message, index) => (
            <div
              className={`message-row ${message.role === 'user' ? 'from-user' : 'from-agent'}`}
              key={`${message.role}-${index}`}
            >
              <div className={`bubble ${message.error ? 'error' : ''}`}>
                <div className="bubble-label">
                  {message.role === 'user' ? `对 ${message.user} 说` : '助手建议'}
                </div>
                <div>{message.content}</div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message-row from-agent">
              <div className="typing-bubble">生成中...</div>
            </div>
          )}
        </section>

        <footer className="input-zone">
          <div className="hotkey-panel">
            <strong>{hotkeyLabel}</strong>
            <span>{prepared ? '已准备，切回聊天窗口后按快捷键' : '先点击准备快捷键分析'}</span>
          </div>

          {screenshotName && (
            <div className="screenshot-note">
              已附加截图：{screenshotName}
              <button onClick={() => {
                setScreenshotBase64('')
                setScreenshotName('')
              }}>
                移除
              </button>
            </div>
          )}

          <div className="input-row">
            <textarea
              rows={2}
              value={input}
              onChange={(event) => {
                setInput(event.target.value)
                setPrepared(false)
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  sendMessage()
                }
              }}
              placeholder={`对 ${currentUser || '当前聊天对象'} 的聊天需求...`}
            />
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(event) => handleScreenshotFile(event.target.files?.[0])}
            />
            <button
              className="icon-button"
              title="上传截图，由后端 Vision LLM 识别"
              onClick={() => fileInputRef.current?.click()}
            >
              图
            </button>
            {isLoading ? (
              <button className="stop-button" onClick={stopGeneration}>
                停止
              </button>
            ) : (
              <button className="send-button" onClick={sendMessage}>
                准备快捷键分析
              </button>
            )}
          </div>
          <div className="input-tip">准备后切回微信/QQ，按 {hotkeyLabel}</div>
        </footer>

        {lastResult && (
          <details className="debug-box">
            <summary>本轮 Agent 调试信息</summary>
            <pre>{JSON.stringify(lastResult, null, 2)}</pre>
          </details>
        )}
      </div>
    </div>
  )
}

export default App
