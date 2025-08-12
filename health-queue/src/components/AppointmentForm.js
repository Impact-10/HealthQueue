import React, { useState } from 'react';

function AppointmentForm() {
  const [form, setForm] = useState({ name: '', email: '', service: '', date: '' });
  const [submitted, setSubmitted] = useState(false);

  const handleChange = (e) => setForm({...form, [e.target.name]: e.target.value});
  const handleSubmit = (e) => {
    e.preventDefault();
    setSubmitted(true);
  };

  if(submitted) {
    return <div className="alert alert-success">Thank you! Your booking is confirmed.</div>;
  }

  return (
    <form onSubmit={handleSubmit} className="p-4 shadow rounded bg-white">
      <div className="mb-3">
        <label className="form-label">Full Name</label>
        <input name="name" value={form.name} onChange={handleChange} className="form-control" required />
      </div>
      <div className="mb-3">
        <label className="form-label">Email Address</label>
        <input type="email" name="email" value={form.email} onChange={handleChange} className="form-control" required />
      </div>
      <div className="mb-3">
        <label className="form-label">Service</label>
        <select name="service" value={form.service} onChange={handleChange} className="form-select" required>
          <option value="">Select a service</option>
          <option>Doctor Appointment</option>
          <option>Pharmacy Refill</option>
          <option>Lab Test</option>
          <option>Vaccination Slot</option>
          <option>Health Check-up</option>
        </select>
      </div>
      <div className="mb-3">
        <label className="form-label">Preferred Date</label>
        <input type="date" name="date" value={form.date} onChange={handleChange} className="form-control" required />
      </div>
      <button className="btn btn-primary">Confirm Booking</button>
    </form>
  );
}

export default AppointmentForm;
