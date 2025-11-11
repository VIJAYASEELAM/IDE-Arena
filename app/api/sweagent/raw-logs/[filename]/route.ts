import { NextRequest, NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ filename: string }> }
) {
  try {
    const { filename } = await params;
    const logsDir = path.join(process.cwd(), "logs");
    const filePath = path.join(logsDir, filename);

    // Check if file exists
    try {
      await fs.access(filePath);
    } catch {
      return NextResponse.json(
        { error: `File ${filename} not found` },
        { status: 404 }
      );
    }

    const content = await fs.readFile(filePath, "utf-8");

    return new NextResponse(content, {
      headers: {
        "Content-Type": "text/plain",
        "Access-Control-Allow-Origin": "*",
      },
    });
  } catch (error: any) {
    console.error("Error reading raw log file:", error);
    return NextResponse.json(
      { error: "Failed to read log file", details: error.message },
      { status: 500 }
    );
  }
}

