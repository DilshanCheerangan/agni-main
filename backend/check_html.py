f = open(r'c:\Users\Tony Stark\Desktop\projects\agni-main\frontend\index.html', 'r', encoding='utf-8')
content = f.read()
f.close()

lines = content.split('\n')
print(f"File size: {len(content)} bytes")
print(f"Total lines: {len(lines)}")
print(f"Script open tags: {content.count('<script')}")
print(f"Script close tags: {content.count('</script>')}")

# Check for common JS syntax issues
import re

# Check if all braces match
opens = content.count('{')
closes = content.count('}')
print(f"Curly braces: {opens} open, {closes} close")

# Check last 5 lines
print("\nLast 5 lines:")
for i, line in enumerate(lines[-5:], len(lines)-4):
    print(f"  {i}: {line[:100]}")

# Check for unclosed HTML comments
comment_opens = len(re.findall(r'<!--', content))
comment_closes = len(re.findall(r'-->', content))
print(f"\nHTML comments: {comment_opens} open, {comment_closes} close")

# Check if there's a JS syntax error by looking for common issues
# Find lines with potential issues  
for i, line in enumerate(lines, 1):
    stripped = line.strip()
    if '`${' in stripped and stripped.count('`') % 2 != 0:
        # Template literal might span multiple lines, skip
        pass
