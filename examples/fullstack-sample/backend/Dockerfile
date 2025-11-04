# Use an official Node.js runtime as a parent image
FROM node:18-alpine

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy package.json and package-lock.json (or yarn.lock)
COPY package*.json ./

# Install project dependencies
RUN npm install
# If you are using yarn:
# RUN yarn install

# Copy the rest of the application code
COPY . .

# Make port 3001 available to the world outside this container
EXPOSE 3001

# Define environment variables (can be overridden)
ENV NODE_ENV=development
ENV PORT=3001

# Run the app when the container launches
CMD ["npm", "start"]
