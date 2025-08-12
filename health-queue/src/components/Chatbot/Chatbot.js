import React, { useState } from 'react';
import ChatMessage from './ChatMessage';
import './chatbot.css';

function Chatbot() {
  const [messages, setMessages] = useState([{ from: 'bot', text: 'Hi! I am HealthBot 🤖. How can I assist you today?' }]);
  const [input, setInput] = useState('');
  const [isOpen, setIsOpen] = useState(false);

  const handleSend = () => {
    if(!input.trim()) return;
    setMessages([...messages, { from: 'user', text: input }]);
    setTimeout(() => {
      setMessages(prev => [...prev, { from: 'bot', text: 'Got it! We will process your request shortly.' }]);
    }, 1000);
    setInput('');
  };

  return (
    <div>
      {isOpen && (
        <div className="chat-window shadow">
          <div className="chat-header">HealthBot Assistant</div>
          <div className="chat-body">
            {messages.map((msg, idx) => <ChatMessage key={idx} {...msg} />)}
          </div>
          <div className="chat-footer d-flex">
            <input value={input} onChange={(e)=>setInput(e.target.value)} className="form-control" placeholder="Type your message..." />
            <button onClick={handleSend} className="btn btn-primary ms-1">Send</button>
          </div>
        </div>
      )}
      <button className="chat-toggle-btn btn btn-primary" onClick={()=>setIsOpen(!isOpen)}>
        <i className="fa fa-comments"></i>
      </button>
    </div>
  );
}

export default Chatbot;
