const mongoose = require('mongoose');

const StudentSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  major: { type: String, required: true },
  // Task 1 ke liye important fields:
  gpa: { type: Number, default: 0.0 },
  creditsCompleted: { type: Number, default: 0 },
  coursesTaken: { type: Number, default: 0 }
});

module.exports = mongoose.model('Student', StudentSchema);