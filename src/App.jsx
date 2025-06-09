import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [todos, setTodos] = useState([]);
  const [newTodo, setNewTodo] = useState('');

  // Placeholder for API functions
  const fetchTodos = async () => {
    // Replace with actual API call
    console.log('Fetching todos...');
    // Example: const response = await axios.get('/api/todos');
    // setTodos(response.data);
    setTodos([
      { id: 1, title: 'Learn React', completed: false },
      { id: 2, title: 'Build a Todo App', completed: true },
    ]);
  };

  const addTodo = async () => {
    if (!newTodo.trim()) return;
    // Replace with actual API call
    console.log('Adding todo:', newTodo);
    // Example: const response = await axios.post('/api/todos', { title: newTodo, completed: false });
    // setTodos([...todos, response.data]);
    const newId = todos.length > 0 ? Math.max(...todos.map(t => t.id)) + 1 : 1;
    setTodos([...todos, { id: newId, title: newTodo, completed: false }]);
    setNewTodo('');
  };

  const toggleComplete = async (id) => {
    // Replace with actual API call
    console.log('Toggling complete for todo:', id);
    // Example: const todoToUpdate = todos.find(t => t.id === id);
    // const response = await axios.put(`/api/todos/${id}`, { ...todoToUpdate, completed: !todoToUpdate.completed });
    // setTodos(todos.map(todo => (todo.id === id ? response.data : todo)));
    setTodos(
      todos.map(todo =>
        todo.id === id ? { ...todo, completed: !todo.completed } : todo
      )
    );
  };

  const deleteTodo = async (id) => {
    // Replace with actual API call
    console.log('Deleting todo:', id);
    // Example: await axios.delete(`/api/todos/${id}`);
    // setTodos(todos.filter(todo => todo.id !== id));
    setTodos(todos.filter(todo => todo.id !== id));
  };

  useEffect(() => {
    fetchTodos();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Todo List</h1>
      </header>
      <div className="add-todo-form">
        <input
          type="text"
          value={newTodo}
          onChange={(e) => setNewTodo(e.target.value)}
          placeholder="Add a new todo"
        />
        <button onClick={addTodo}>Add Todo</button>
      </div>
      <ul className="todo-list">
        {todos.map(todo => (
          <li key={todo.id} className={`todo-item ${todo.completed ? 'completed' : ''}`}>
            <span onClick={() => toggleComplete(todo.id)}>
              {todo.title}
            </span>
            <button onClick={() => deleteTodo(todo.id)}>Delete</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
