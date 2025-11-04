import React, { useState, useEffect } from 'react';
import axios from 'axios';
import TodoList from './components/TodoList';
import AddTodoForm from './components/AddTodoForm';
import './App.css';

function App() {
  const [todos, setTodos] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchTodos();
  }, []);

  const fetchTodos = async () => {
    try {
      const response = await axios.get('/api/todos');
      setTodos(response.data);
      setError('');
    } catch (err) {
      console.error("Error fetching todos:", err);
      setError('Failed to fetch todos. Is the backend server running?');
      setTodos([]); // Clear todos on error
    }
  };

  const addTodo = async (text) => {
    if (!text.trim()) {
      setError('Todo text cannot be empty.');
      return;
    }
    try {
      const response = await axios.post('/api/todos', { text });
      setTodos([...todos, response.data]);
      setError('');
    } catch (err) {
      console.error("Error adding todo:", err);
      setError('Failed to add todo.');
    }
  };

  const toggleComplete = async (id) => {
    try {
      const todoToUpdate = todos.find(todo => todo.id === id);
      if (!todoToUpdate) return;

      const updatedTodo = { ...todoToUpdate, completed: !todoToUpdate.completed };
      const response = await axios.put(`/api/todos/${id}`, updatedTodo);
      setTodos(todos.map(todo => (todo.id === id ? response.data : todo)));
      setError('');
    } catch (err) {
      console.error("Error toggling todo completion:", err);
      setError('Failed to update todo.');
    }
  };

  const deleteTodo = async (id) => {
    try {
      await axios.delete(`/api/todos/${id}`);
      setTodos(todos.filter(todo => todo.id !== id));
      setError('');
    } catch (err) {
      console.error("Error deleting todo:", err);
      setError('Failed to delete todo.');
    }
  };

  return (
    <div className="app-container">
      <h1>My To-Do List</h1>
      {error && <p className="error-message">{error}</p>}
      <AddTodoForm onAddTodo={addTodo} />
      <TodoList
        todos={todos}
        onToggleComplete={toggleComplete}
        onDeleteTodo={deleteTodo}
      />
    </div>
  );
}

export default App;
