import React from 'react';

function Footer() {
  return (
    <footer>
      <div className="container">
        <p>&copy; {new Date().getFullYear()} HealthBot. All Rights Reserved.</p>
      </div>
    </footer>
  );
}

export default Footer;
