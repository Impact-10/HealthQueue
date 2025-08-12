import React from 'react';
import ServiceCard from '../components/ServiceCard';

function Services() {
  const services = [
    { icon: 'fa-user-md', title: 'Doctor Appointment', description: 'Book consultations with trusted doctors.' },
    { icon: 'fa-pills', title: 'Pharmacy Refill', description: 'Order medicines and schedule pickups.' },
    { icon: 'fa-flask', title: 'Lab Test', description: 'Schedule diagnostic & health lab tests.' },
    { icon: 'fa-syringe', title: 'Vaccination Slot', description: 'Book vaccine appointments easily.' },
    { icon: 'fa-heartbeat', title: 'Health Check-up', description: 'Get regular preventive health check-ups.' }
  ];

  return (
    <section className="container my-5">
      <div className="row">
        {services.map((s, idx) => <ServiceCard key={idx} {...s} />)}
      </div>
    </section>
  );
}

export default Services;
