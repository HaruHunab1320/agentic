const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const port = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// In-memory store for To-Do items
let todos = [
  { id: 1, text: 'Learn React', completed: false },
  { id: 2, text: 'Build a To-Do App', completed: false },
];
let nextId = 3;

// API Routes

// Get all To-Do items
app.get('/todos', (req, res) => {
  res.json(todos);
});

// Add a new To-Do item
app.post('/todos', (req, res) => {
  const { text } = req.body;
  if (!text) {
    return res.status(400).json({ error: 'Text is required' });
  }
  const newTodo = {
    id: nextId++,
    text,
    completed: false,
  };
  todos.push(newTodo);
  res.status(201).json(newTodo);
});

// Update a To-Do item
app.put('/todos/:id', (req, res) => {
  const { id } = req.params;
  const { text, completed } = req.body;
  const todoIndex = todos.findIndex(todo => todo.id === parseInt(id));

  if (todoIndex === -1) {
    return res.status(404).json({ error: 'Todo not found' });
  }

  const updatedTodo = { ...todos[todoIndex] };
  if (text !== undefined) {
    updatedTodo.text = text;
  }
  if (completed !== undefined) {
    updatedTodo.completed = completed;
  }

  todos[todoIndex] = updatedTodo;
  res.json(updatedTodo);
});

// Delete a To-Do item
app.delete('/todos/:id', (req, res) => {
  const { id } = req.params;
  const todoIndex = todos.findIndex(todo => todo.id === parseInt(id));

  if (todoIndex === -1) {
    return res.status(404).json({ error: 'Todo not found' });
  }

  todos.splice(todoIndex, 1);
  res.status(204).send(); // No content
});

app.listen(port, () => {
  console.log(`Backend server is running on http://localhost:${port}`);
});
