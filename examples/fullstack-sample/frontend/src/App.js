import React from 'react';
import './App.css'; // You might want to create a basic App.css or remove this line

function App() {
  // Example of fetching data from the backend:
  // const [message, setMessage] = React.useState('');
  // React.useEffect(() => {
  //   fetch('http://localhost:5000/api/greeting') // Ensure backend is running and CORS is configured
  //     .then(res => res.json())
  //     .then(data => setMessage(data.message))
  //     .catch(err => console.error("Failed to fetch greeting:", err));
  // }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Welcome to the React Frontend</h1>
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        {/* <p>Backend says: {message || "Loading..."}</p> */}
      </header>
    </div>
  );
}

// A simple App.css could be:
/*
.App {
  text-align: center;
}

.App-header {
  background-color: #282c34;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-size: calc(10px + 2vmin);
  color: white;
}
*/

export default App;
