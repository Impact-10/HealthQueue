import React from 'react';
import { Link } from 'react-router-dom';

function Home() {
  return (
    <div>
      <section className="hero-section text-center">
        <div className="container">
          <h1 className="display-4 fw-bold">Healthcare Appointment & Service Ticketing Bot</h1>
          <p className="lead mt-3">Book doctor visits, pharmacy refills, lab tests & more instantly!</p>
          <Link to="/services" className="btn btn-light btn-lg mt-4">Get Started</Link>
        </div>
      </section>
    </div>
  );
}

export default Home;
