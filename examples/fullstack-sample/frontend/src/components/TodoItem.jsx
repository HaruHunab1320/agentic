import React from 'react';

function TodoItem({ todo, onToggleComplete, onDeleteTodo }) {
  return (
    <li>
      <span
        className={todo.completed ? 'completed' : ''}
        onClick={() => onToggleComplete(todo._id)} // Assuming MongoDB _id
        style={{ textDecoration: todo.completed ? 'line-through' : 'none', cursor: 'pointer' }}
      >
        {todo.text}
      </span>
      <button onClick={() => onDeleteTodo(todo._id)}>Delete</button> {/* Assuming MongoDB _id */}
    </li>
  );
}

export default TodoItem;
