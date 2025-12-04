const express = require('express');
const router = express.Router();
const studentController = require('../controllers/studentController');

router.post('/', studentController.createStudent);
router.get('/', studentController.getStudents);
router.post('/:id/calculate-gpa', studentController.calculateGPA);

module.exports = router;