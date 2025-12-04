import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [students, setStudents] = useState([]);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [major, setMajor] = useState('');

  // Load students when page opens
  useEffect(() => {
    fetchStudents();
  }, []);

  const fetchStudents = async () => {
    try {
      // This calls your Backend API
      const res = await axios.get('/api/students');
      setStudents(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    // Send new student to Backend
    await axios.post('/api/students', { name, email, major });
    // Refresh the list
    fetchStudents();
    setName(''); setEmail(''); setMajor('');
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h1>University Student Manager</h1>
      
      {/* Form to Add Student */}
      <div style={{ marginBottom: '20px', padding: '10px', border: '1px solid #ccc' }}>
        <h3>Add New Student</h3>
        <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px' }}>
          <input 
            placeholder="Name" 
            value={name} 
            onChange={e => setName(e.target.value)} 
            required 
          />
          <input 
            placeholder="Email" 
            value={email} 
            onChange={e => setEmail(e.target.value)} 
            required 
          />
          <input 
            placeholder="Major" 
            value={major} 
            onChange={e => setMajor(e.target.value)} 
            required 
          />
          <button type="submit" style={{ cursor: 'pointer' }}>Add Student</button>
        </form>
      </div>

      {/* List of Students */}
      <h3>Student Directory</h3>
      <ul>
        {students.length === 0 ? <p>No students found. Add one!</p> : null}
        {students.map(s => (
          <li key={s._id} style={{ marginBottom: '5px' }}>
            <strong>{s.name}</strong> ({s.major}) - GPA: {s.gpa ? s.gpa.toFixed(2) : '0.00'}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;