const express = require('express');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json()); // Replaces body-parser for JSON

// In-memory store for To-Do items
let todos = [
  { id: uuidv4(), text: 'Learn Node.js', completed: false },
  { id: uuidv4(), text: 'Build a To-Do App', completed: false },
  { id: uuidv4(), text: 'Connect Frontend to Backend', completed: true },
];

// API Routes

// Get all todos
app.get('/todos', (req, res) => {
  res.status(200).json(todos);
});

// Create a new todo
app.post('/todos', (req, res) => {
  const { text } = req.body;
  if (!text) {
    return res.status(400).json({ message: 'Text is required for a new todo' });
  }
  const newTodo = {
    id: uuidv4(),
    text,
    completed: false,
  };
  todos.push(newTodo);
  res.status(201).json(newTodo);
});

// Get a single todo by id (Optional, but good practice)
app.get('/todos/:id', (req, res) => {
  const { id } = req.params;
  const todo = todos.find(t => t.id === id);
  if (todo) {
    res.status(200).json(todo);
  } else {
    res.status(404).json({ message: 'Todo not found' });
  }
});

// Update a todo
app.put('/todos/:id', (req, res) => {
  const { id } = req.params;
  const { text, completed } = req.body;

  const todoIndex = todos.findIndex(t => t.id === id);

  if (todoIndex === -1) {
    return res.status(404).json({ message: 'Todo not found' });
  }

  // Update fields if they are provided
  if (text !== undefined) {
    todos[todoIndex].text = text;
  }
  if (completed !== undefined) {
    todos[todoIndex].completed = completed;
  }

  res.status(200).json(todos[todoIndex]);
});

// Delete a todo
app.delete('/todos/:id', (req, res) => {
  const { id } = req.params;
  const initialLength = todos.length;
  todos = todos.filter(t => t.id !== id);

  if (todos.length < initialLength) {
    res.status(204).send(); // No content, successful deletion
  } else {
    res.status(404).json({ message: 'Todo not found' });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
