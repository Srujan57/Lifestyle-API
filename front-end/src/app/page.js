'use client'
import { useState, useRef, useEffect } from "react";

export default function Home() {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hi! I'm your health coach. How can I help you today?" }
  ]);
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [city, setCity] = useState("Columbus");
  const [fitbitConnected, setFitbitConnected] = useState(false);

  const bottomRef = useRef(null);

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Check if Fitbit was just connected (redirect from callback)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("fitbit") === "connected") {
      setFitbitConnected(true);
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: "Fitbit connected successfully! I can now use your activity data to give more personalized recommendations." }
      ]);
      // Clean up the URL
      window.history.replaceState({}, document.title, "/");
    }
  }, []);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userText = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: userText }]);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/endpoint`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ message: userText, history, city }),
      });

      const data = await response.json();

      if (data.reply) {
        setMessages(prev => [...prev, { role: "assistant", content: data.reply }]);
        setHistory(data.history);
      } else {
        setMessages(prev => [
          ...prev,
          { role: "assistant", content: data.error || "Something went wrong. Please try again." },
        ]);
      }
    } catch (err) {
      console.error(err);
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: "Connection error. Is the backend running?" },
      ]);
    }

    setLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const connectFitbit = () => {
    window.location.href = `${API_BASE_URL}/authorize`;
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 font-sans">
      {/* Header */}
      <div className="bg-white border-b px-6 py-4 flex items-center justify-between shadow-sm">
        <div>
          <h1 className="text-lg font-semibold text-gray-800">Health Coach</h1>
          <p className="text-xs text-gray-400">Powered by your literature</p>
        </div>
        <div className="flex items-center gap-3">
          <input
            className="text-sm border rounded px-2 py-1 w-32 text-gray-600"
            value={city}
            onChange={(e) => setCity(e.target.value)}
            placeholder="City for weather"
          />
          <button
            onClick={connectFitbit}
            className={`text-sm px-3 py-1 rounded transition ${
              fitbitConnected
                ? "bg-green-100 text-green-700"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            {fitbitConnected ? "Fitbit Connected" : "Connect Fitbit"}
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-xl px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-blue-500 text-white rounded-br-sm"
                  : "bg-white text-gray-800 border rounded-bl-sm shadow-sm"
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border rounded-2xl rounded-bl-sm px-4 py-3 text-sm text-gray-400 shadow-sm">
              Thinking...
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="bg-white border-t px-4 py-4 flex gap-3 items-end">
        <textarea
          className="flex-1 border rounded-xl px-4 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-300"
          rows={2}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about exercise, goals, or your health..."
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          className="bg-blue-500 text-white px-5 py-2 rounded-xl text-sm hover:bg-blue-600 transition disabled:opacity-40"
        >
          Send
        </button>
      </div>
    </div>
  );
}
