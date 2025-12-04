const Student = require('../models/Student');

// 1. Create Student
exports.createStudent = async (req, res) => {
  try {
    const student = new Student(req.body);
    await student.save();
    res.status(201).json(student);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
};

// 2. Get Students
exports.getStudents = async (req, res) => {
  try {
    const students = await Student.find();
    res.json(students);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// 3. Calculate GPA (BUG HAI ISME)
// Wrong Logic: Dividing by 'coursesTaken' instead of 'creditsCompleted'
exports.calculateGPA = async (req, res) => {
  try {
    const { id } = req.params;
    const { totalGradePoints } = req.body;
    
    const student = await Student.findById(id);
    if (!student) return res.status(404).json({ error: "Student not found" });

    // --- BUG START ---
    if (student.coursesTaken > 0) {
        // Logic Galat Hai: Hamein credits se divide karna chahiye tha
        student.gpa = totalGradePoints / student.coursesTaken; 
    } else {
        student.gpa = 0;
    }
    // --- BUG END ---
    
    await student.save();
    res.json({ gpa: student.gpa });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};