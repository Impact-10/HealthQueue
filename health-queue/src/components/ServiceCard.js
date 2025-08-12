import React from 'react';
import { Link } from 'react-router-dom';

function ServiceCard({ icon, title, description }) {
  return (
    <div className="col-md-4 mb-4">
      <div className="card h-100 shadow-sm text-center p-3">
        <div className="mb-3">
          <i className={`fa ${icon} fa-3x text-primary`}></i>
        </div>
        <h5 className="card-title">{title}</h5>
        <p className="card-text">{description}</p>
        <Link to="/appointment" className="btn btn-primary">Book Now</Link>
      </div>
    </div>
  );
}

export default ServiceCard;
