import React from 'react';
import TodoItem from './TodoItem';

function TodoList({ todos, onToggleComplete, onDeleteTodo }) {
  if (!todos || todos.length === 0) {
    return <p>No todos yet. Add one!</p>;
  }

  return (
    <ul>
      {todos.map(todo => (
        <TodoItem
          key={todo._id} // Assuming MongoDB _id, adjust if your ID field is different
          todo={todo}
          onToggleComplete={onToggleComplete}
          onDeleteTodo={onDeleteTodo}
        />
      ))}
    </ul>
  );
}

export default TodoList;
