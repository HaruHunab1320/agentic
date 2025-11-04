import React from 'react';

function TodoItem({ todo, onToggleComplete, onDeleteTodo }) {
  return (
    <li className="todo-item">
      <span
        className={`todo-item-toggle ${todo.completed ? 'completed' : ''}`}
        onClick={() => onToggleComplete(todo.id)}
        role="button"
        tabIndex={0}
        aria-pressed={todo.completed}
        aria-label={todo.completed ? 'Mark as incomplete' : 'Mark as complete'}
      >
      </span>
      <span
        className={`todo-item-text ${todo.completed ? 'completed' : ''}`}
        onClick={() => onToggleComplete(todo.id)}
        role="button"
        tabIndex={0}
      >
        {todo.text}
      </span>
      <div className="todo-item-actions">
        <button onClick={() => onDeleteTodo(todo.id)} aria-label={`Delete todo: ${todo.text}`}>
          Delete
        </button>
      </div>
    </li>
  );
}

export default TodoItem;
