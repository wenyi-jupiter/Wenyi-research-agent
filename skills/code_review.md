---
name: code_review
description: Perform thorough code review with best practices analysis
version: 1.0.0
tags:
  - development
  - review
  - quality
parameters:
  - name: code
    type: string
    required: true
    description: The code to review
  - name: language
    type: string
    required: false
    description: Programming language (auto-detected if not provided)
  - name: focus_areas
    type: array
    required: false
    description: Specific areas to focus on (security, performance, style)
---

# Code Review Skill

Perform comprehensive code reviews following industry best practices.

## Instructions

When reviewing code, analyze the following aspects:

### 1. Code Quality
- Check for clear, descriptive naming conventions
- Verify proper code organization and structure
- Look for code duplication (DRY principle)
- Assess readability and maintainability

### 2. Security
- Check for input validation
- Look for potential injection vulnerabilities
- Verify proper authentication/authorization
- Check for sensitive data exposure

### 3. Performance
- Identify potential bottlenecks
- Check for unnecessary computations
- Look for memory leaks
- Assess algorithm complexity

### 4. Best Practices
- Verify error handling
- Check for proper logging
- Assess test coverage
- Look for documentation

## Output Format

Provide feedback in the following format:

```
## Summary
[Brief overall assessment]

## Issues Found
- [Severity: HIGH/MEDIUM/LOW] [Issue description]
  - Location: [line/function]
  - Suggestion: [how to fix]

## Positive Aspects
- [What's done well]

## Recommendations
1. [Priority recommendation]
2. [Additional suggestion]
```

## Examples

- Review a Python function for security vulnerabilities
- Analyze a JavaScript module for performance issues
- Check a SQL query for injection risks
