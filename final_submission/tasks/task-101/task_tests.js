const request = require('supertest');
const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');
const app = require('../../repo/server');
const Student = require('../../repo/models/Student');

let mongoServer;

beforeAll(async () => {
  mongoServer = await MongoMemoryServer.create();
  await mongoose.connect(mongoServer.getUri());
});

afterAll(async () => {
  await mongoose.disconnect();
  await mongoServer.stop();
});

describe('Task 101: GPA Fix', () => {
  it('should calculate GPA using CREDITS, not course count', async () => {
    const student = new Student({
      name: 'Test Student',
      email: 'test@student.com',
      major: 'CS',
      creditsCompleted: 20,
      coursesTaken: 4
    });
    await student.save();

    // Sending 80 grade points.
    // If logic is WRONG (dividing by 4 courses), GPA = 20.0 (Fail)
    // If logic is CORRECT (dividing by 20 credits), GPA = 4.0 (Pass)
    const res = await request(app)
      .post(`/api/students/${student._id}/calculate-gpa`)
      .send({ totalGradePoints: 80 });

    expect(res.body.gpa).toBe(4.0);
  });
});