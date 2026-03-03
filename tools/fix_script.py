import os

file_path = "d:/SignVision_S8_V2/web/webrtc/script.js"

# Read the file
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

# Find the last good line (around 522)
# We look for the disconnect handler end
clean_lines = []
for i, line in enumerate(lines):
    if "socket.on('disconnect'" in line:
        # Keep this and the next few lines until the closing brace
        clean_lines.append(line)
        # We know it ends with }); around 3 lines later
        continue
    
    # Check for corruption (wide spacing)
    if "  / /   " in line:
        break
        
    clean_lines.append(line)

# Ensure we have the basic structure (up to line 522 roughly)
# But essentially just stripping the bad part is enough if we caught the break.
# Let's be safer: simply take lines 0 to 522 based on the view_file output we just saw.
# The view_file output showed 522 lines of good code.

final_lines = lines[:522]

new_code = """
// --- Text-to-Sign Search ---
async function searchAndPlaySign() {
    const input = document.getElementById('signSearchInput');
    const word = input.value.trim();
    if (!word) return;

    const container = document.getElementById('signPlayerContainer');
    const video = document.getElementById('signVideo');
    // Ensure label exists or fallback
    let label = container.querySelector('.player-label');
    if (!label) {
         // Create if missing (though it is in index.html)
         label = document.createElement('div');
         label.className = 'player-label';
         container.appendChild(label);
    }

    label.innerText = `Searching '${word}'...`;
    container.classList.remove('hidden');
    container.classList.add('show');
    
    try {
        const response = await fetch(`/search_sign?word=${encodeURIComponent(word)}`);
        
        if (response.ok) {
            const data = await response.json();
             if (data.video_url) {
                label.innerText = `Sign for: ${data.word}`;
                video.src = data.video_url;
                video.play().catch(e => console.error("Autoplay error:", e));
             }
        } else {
             label.innerText = `Sign '${word}' not found`;
             setTimeout(() => {
                 container.classList.add('hidden');
                 container.classList.remove('show');
             }, 3000);
        }
    } catch (e) {
        console.error(e);
        label.innerText = "Error searching.";
    }
}

window.searchAndPlaySign = searchAndPlaySign;

// Bind Enter key
const searchInput = document.getElementById('signSearchInput');
if (searchInput) {
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchAndPlaySign();
    });
}
"""

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(final_lines)
    f.write(new_code)

print("Fixed script.js")
