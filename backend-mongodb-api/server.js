const express = require('express');
const cors = require('cors');
const { MongoClient, ObjectId } = require('mongodb');

const app = express();
const port = 5000;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB connection URI and client
const uri = 'mongodb://localhost:27017';
const client = new MongoClient(uri);

let historyCollection;

async function connectDB() {
  try {
    await client.connect();
    const database = client.db('historyDB');
    historyCollection = database.collection('history');
    console.log('Connected to MongoDB');
  } catch (error) {
    console.error('MongoDB connection error:', error);
  }
}

// Routes

// Get all history items
app.get('/history', async (req, res) => {
  try {
    const historyItems = await historyCollection.find({}).toArray();
    res.json(historyItems);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch history' });
  }
});

// Add a new history item
app.post('/history', async (req, res) => {
  try {
    const newItem = req.body;
    const result = await historyCollection.insertOne(newItem);
    res.status(201).json({ _id: result.insertedId, ...newItem });
  } catch (error) {
    res.status(500).json({ error: 'Failed to add history item' });
  }
});

// Delete a history item by id
app.delete('/history/:id', async (req, res) => {
  try {
    const id = req.params.id;
    const result = await historyCollection.deleteOne({ _id: new ObjectId(id) });
    if (result.deletedCount === 1) {
      res.json({ message: 'History item deleted' });
    } else {
      res.status(404).json({ error: 'History item not found' });
    }
  } catch (error) {
    res.status(500).json({ error: 'Failed to delete history item' });
  }
});

// Start server and connect to DB
app.listen(port, () => {
  console.log(`MongoDB API server running on port ${port}`);
  connectDB();
});
