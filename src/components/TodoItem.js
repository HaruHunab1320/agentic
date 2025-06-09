import React from 'react';

function TodoItem({ todo, toggleComplete, deleteTodo }) {
  return (
    <li className="todo-item">
      <input
        type="checkbox"
        checked={todo.completed}
        onChange={() => toggleComplete(todo.id)}
      />
      <span
        className={todo.completed ? 'completed' : ''}
        onClick={() => toggleComplete(todo.id)} // Allow toggling by clicking text
      >
        {todo.text}
      </span>
      <button onClick={() => deleteTodo(todo.id)}>Delete</button>
    </li>
  );
}

export default TodoItem;
