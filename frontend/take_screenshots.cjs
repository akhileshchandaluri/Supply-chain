const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const UI_DIR = path.resolve(__dirname, '../data/ui');
if (!fs.existsSync(UI_DIR)) {
  fs.mkdirSync(UI_DIR, { recursive: true });
}

const PAGES = [
  "dashboard",
  "demand",
  "risk",
  "anomaly",
  "routing",
  "rl"
];

(async () => {
  try {
    console.log("Starting puppeteer...");
    const browser = await puppeteer.launch({
      headless: "new",
    });
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900 });

    console.log("Navigating to http://localhost:5173 ...");
    // Wait until network is idle so we know frontend is fully loaded
    await page.goto('http://localhost:5173', { waitUntil: 'networkidle0' });
    
    // Give it an extra second to mount
    await new Promise(r => setTimeout(r, 1000));

    console.log("Clicking 'Run Live Order' to populate data...");
    const liveOrderBtn = await page.$('.btn-primary');
    if (liveOrderBtn) {
        await liveOrderBtn.click();
        // wait for the simulated order to process (simulateOrder has a slight delay)
        await new Promise(r => setTimeout(r, 2000)); 
    }

    const navItems = await page.$$('.nav-item');
    if (navItems.length !== PAGES.length) {
        console.warn(`Expected ${PAGES.length} nav items, found ${navItems.length}`);
    }

    for (let i = 0; i < PAGES.length; i++) {
      const pid = PAGES[i];
      console.log(`Processing ${pid}...`);
      
      if (i < navItems.length) {
          await navItems[i].click();
          // Wait for page transition (App.jsx has a 0.32s transition) + data load
          await new Promise(r => setTimeout(r, 1000));
      }

      const screenshotPath = path.join(UI_DIR, `${pid}.png`);
      await page.screenshot({ path: screenshotPath, fullPage: true });
      console.log(`Saved ${screenshotPath}`);
    }

    await browser.close();
    console.log("Done!");
  } catch (err) {
    console.error("Error:", err);
    process.exit(1);
  }
})();
