import React, { useState } from 'react';

function Contact() {
  const [sent, setSent] = useState(false);
  
  const handleSubmit = (e) => {
    e.preventDefault();
    setSent(true);
  };

  return (
    <section className="container my-5">
      <h2>Contact Us</h2>
      {sent ? (
        <div className="alert alert-success">Message sent successfully!</div>
      ) : (
        <form onSubmit={handleSubmit} className="bg-white p-4 shadow rounded">
          <div className="mb-3">
            <label className="form-label">Name</label>
            <input className="form-control" required />
          </div>
          <div className="mb-3">
            <label className="form-label">Email</label>
            <input type="email" className="form-control" required />
          </div>
          <div className="mb-3">
            <label className="form-label">Message</label>
            <textarea className="form-control" rows={4} required></textarea>
          </div>
          <button className="btn btn-primary">Send</button>
        </form>
      )}
    </section>
  );
}

export default Contact;
