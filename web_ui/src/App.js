import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Send,
    Plus,
    MessageSquare,
    BookOpen,
    Calendar,
    Settings,
    Sparkles,
    User,
    Bot,
    Menu,
    X,
    ChevronRight
} from 'lucide-react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './styles/App.css';

function App() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId, setSessionId] = useState('');
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [showWelcome, setShowWelcome] = useState(true);
    const [showSettings, setShowSettings] = useState(false);
    const [temperature, setTemperature] = useState(() => {
        return parseFloat(localStorage.getItem('hkbu_temperature')) || 0.7;
    });
    const [maxTokens, setMaxTokens] = useState(() => {
        return parseInt(localStorage.getItem('hkbu_max_tokens')) || 512;
    });
    const messagesEndRef = useRef(null);

    useEffect(() => {
        // Initialize session
        const savedSession = localStorage.getItem('hkbu_session_id');
        if (savedSession) {
            setSessionId(savedSession);
        } else {
            const newSession = uuidv4();
            setSessionId(newSession);
            localStorage.setItem('hkbu_session_id', newSession);
        }
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage = input.trim();
        setInput('');
        setShowWelcome(false);

        // Add user message
        setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
        setIsLoading(true);

        try {
            const response = await axios.post('/api/chat', {
                message: userMessage,
                session_id: sessionId,
                temperature: temperature,
                max_tokens: maxTokens
            });

            // Add assistant message
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.data.response,
                type: response.data.type
            }]);
        } catch (error) {
            console.error('Error:', error);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Sorry, I encountered an error. Please try again.',
                type: 'error'
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const startNewChat = () => {
        const newSession = uuidv4();
        setSessionId(newSession);
        localStorage.setItem('hkbu_session_id', newSession);
        setMessages([]);
        setShowWelcome(true);
    };

    const suggestionCards = [
        { icon: BookOpen, text: "What programs does HKBU offer?", color: "#3b82f6" },
        { icon: Calendar, text: "Create a study plan for me", color: "#10b981" },
        { icon: MessageSquare, text: "What is academic advising?", color: "#f59e0b" },
        { icon: Sparkles, text: "Tell me about student life", color: "#8b5cf6" },
    ];

    return (
        <div className="app">
            {/* Sidebar */}
            <AnimatePresence>
                {sidebarOpen && (
                    <motion.aside
                        className="sidebar"
                        initial={{ x: -280 }}
                        animate={{ x: 0 }}
                        exit={{ x: -280 }}
                        transition={{ duration: 0.3 }}
                    >
                        <div className="sidebar-header">
                            <div className="logo">
                                <div className="logo-icon">
                                    <Sparkles size={24} />
                                </div>
                                <span className="logo-text">HKBU Assistant</span>
                            </div>
                            <button
                                className="sidebar-close-btn"
                                onClick={() => setSidebarOpen(false)}
                                title="Close sidebar"
                            >
                                <X size={20} />
                            </button>
                        </div>

                        <button className="new-chat-btn" onClick={startNewChat}>
                            <Plus size={18} />
                            <span>New Chat</span>
                        </button>

                        <div className="sidebar-section">
                            <h3>Quick Actions</h3>
                            <div className="quick-actions">
                                <button className="action-btn" onClick={() => { setInput("Create a study plan"); }}>
                                    <Calendar size={16} />
                                    <span>Study Plan</span>
                                </button>
                                <button className="action-btn" onClick={() => { setInput("What academic programs are available?"); }}>
                                    <BookOpen size={16} />
                                    <span>Programs</span>
                                </button>
                            </div>
                        </div>

                        <div className="sidebar-footer">
                            <div className="session-info">
                                <span>Session ID:</span>
                                <code>{sessionId.slice(0, 8)}...</code>
                            </div>
                        </div>
                    </motion.aside>
                )}
            </AnimatePresence>

            {/* Sidebar Toggle Button (visible when sidebar is closed) */}
            {!sidebarOpen && (
                <motion.button
                    className="sidebar-toggle-btn"
                    onClick={() => setSidebarOpen(true)}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    title="Open sidebar"
                >
                    <Menu size={20} />
                </motion.button>
            )}

            {/* Main Content */}
            <main className={`main-content ${!sidebarOpen ? 'sidebar-closed' : ''}`}>
                {/* Header */}
                <header className="header">
                    {!sidebarOpen && <div className="header-spacer" />}
                    <h1 className="header-title">HKBU Course Assistant</h1>
                    <div className="header-actions">
                        <button
                            className="icon-btn"
                            title="Settings"
                            onClick={() => setShowSettings(true)}
                        >
                            <Settings size={20} />
                        </button>
                    </div>
                </header>

                {/* Settings Modal */}
                <AnimatePresence>
                    {showSettings && (
                        <motion.div
                            className="settings-overlay"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setShowSettings(false)}
                        >
                            <motion.div
                                className="settings-panel"
                                initial={{ scale: 0.9, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                exit={{ scale: 0.9, opacity: 0 }}
                                onClick={e => e.stopPropagation()}
                            >
                                <div className="settings-header">
                                    <h2>Generation Settings</h2>
                                    <button
                                        className="settings-close"
                                        onClick={() => setShowSettings(false)}
                                    >
                                        <X size={20} />
                                    </button>
                                </div>

                                <div className="settings-content">
                                    <div className="setting-item">
                                        <label>
                                            <span>Temperature</span>
                                            <span className="setting-value">{temperature.toFixed(1)}</span>
                                        </label>
                                        <input
                                            type="range"
                                            min="0"
                                            max="2"
                                            step="0.1"
                                            value={temperature}
                                            onChange={(e) => {
                                                const val = parseFloat(e.target.value);
                                                setTemperature(val);
                                                localStorage.setItem('hkbu_temperature', val);
                                            }}
                                        />
                                        <p className="setting-desc">
                                            Lower = more focused, Higher = more creative
                                        </p>
                                    </div>

                                    <div className="setting-item">
                                        <label>
                                            <span>Max Output Length</span>
                                            <span className="setting-value">{maxTokens} tokens</span>
                                        </label>
                                        <input
                                            type="range"
                                            min="128"
                                            max="2048"
                                            step="64"
                                            value={maxTokens}
                                            onChange={(e) => {
                                                const val = parseInt(e.target.value);
                                                setMaxTokens(val);
                                                localStorage.setItem('hkbu_max_tokens', val);
                                            }}
                                        />
                                        <p className="setting-desc">
                                            Maximum number of tokens to generate
                                        </p>
                                    </div>
                                </div>

                                <div className="settings-footer">
                                    <p>Settings are saved automatically</p>
                                </div>
                            </motion.div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Chat Area */}
                <div className="chat-container">
                    {showWelcome && messages.length === 0 ? (
                        <div className="welcome-screen">
                            <motion.div
                                className="welcome-content"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.5 }}
                            >
                                <div className="welcome-icon">
                                    <Sparkles size={48} />
                                </div>
                                <h2>Welcome to HKBU Assistant</h2>
                                <p>Your AI-powered guide to Hong Kong Baptist University</p>

                                <div className="suggestion-grid">
                                    {suggestionCards.map((card, index) => (
                                        <motion.button
                                            key={index}
                                            className="suggestion-card"
                                            onClick={() => { setInput(card.text); }}
                                            initial={{ opacity: 0, y: 20 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: index * 0.1 }}
                                            whileHover={{ scale: 1.02, y: -2 }}
                                            whileTap={{ scale: 0.98 }}
                                        >
                                            <div className="suggestion-icon" style={{ backgroundColor: `${card.color}20`, color: card.color }}>
                                                <card.icon size={24} />
                                            </div>
                                            <span>{card.text}</span>
                                            <ChevronRight size={16} className="suggestion-arrow" />
                                        </motion.button>
                                    ))}
                                </div>
                            </motion.div>
                        </div>
                    ) : (
                        <div className="messages">
                            {messages.map((msg, index) => (
                                <motion.div
                                    key={index}
                                    className={`message ${msg.role}`}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    <div className="message-avatar">
                                        {msg.role === 'user' ? (
                                            <div className="avatar user">
                                                <User size={18} />
                                            </div>
                                        ) : (
                                            <div className="avatar assistant">
                                                <Bot size={18} />
                                            </div>
                                        )}
                                    </div>
                                    <div className="message-content">
                                        {msg.role === 'assistant' ? (
                                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                                {msg.content}
                                            </ReactMarkdown>
                                        ) : (
                                            <p>{msg.content}</p>
                                        )}
                                    </div>
                                </motion.div>
                            ))}
                            {isLoading && (
                                <motion.div
                                    className="message assistant loading"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                >
                                    <div className="message-avatar">
                                        <div className="avatar assistant">
                                            <Bot size={18} />
                                        </div>
                                    </div>
                                    <div className="message-content">
                                        <div className="typing-indicator">
                                            <span></span>
                                            <span></span>
                                            <span></span>
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                {/* Input Area */}
                <div className="input-area">
                    <div className="input-container">
                        <textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="Ask anything about HKBU..."
                            rows={1}
                            disabled={isLoading}
                        />
                        <button
                            className={`send-btn ${input.trim() ? 'active' : ''}`}
                            onClick={handleSend}
                            disabled={!input.trim() || isLoading}
                        >
                            <Send size={20} />
                        </button>
                    </div>
                    <p className="input-footer">HKBU Assistant may produce inaccurate information. Please verify important details.</p>
                </div>
            </main>
        </div>
    );
}

export default App;
