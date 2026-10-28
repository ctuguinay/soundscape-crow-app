/* CROW: Community Recording Online Workflow */

'use strict';

const { app, BrowserWindow, dialog, ipcMain, net, shell } = require('electron');
const { execFile, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const util = require('util');
const audiomoth = require('audiomoth-hid');
const { SerialPort } = require('serialport');

// AudioMoth Configuration Settings

const GAIN_MEDIUM = 2;
const SLEEP_DURATION_S = 10 * 60;
const RECORD_DURATION_S = 50 * 60;
const LED_ENABLED = true;
const BATTERY_LEVEL_ENABLED = true;
const RECORDING_PERIODS = [[0, 24 * 60]];
const TIMEZONE_OFFSET_MINS = 0;

const SAMPLE_RATE_CONFIGURATION = { clockDivider: 4, acquisitionCycles: 16, oversampleRate: 1, sampleRate: 384000, sampleRateDivider: 2 };

// Configure AudioMoth Packet

const PACKET_LENGTH = 62;
const MAX_RECORDING_PERIODS = 5;
const PRERECORDING_PREP_TIME = 2;

function buildPacket (sendTime) {

    const packet = Buffer.alloc(PACKET_LENGTH);
    const c = SAMPLE_RATE_CONFIGURATION;
    let i = 0;

    i = packet.writeUInt32LE(sendTime, i);
    i = packet.writeUInt8(GAIN_MEDIUM, i);
    i = packet.writeUInt8(c.clockDivider, i);
    i = packet.writeUInt8(c.acquisitionCycles, i);
    i = packet.writeUInt8(c.oversampleRate, i);
    i = packet.writeUInt32LE(c.sampleRate, i);
    i = packet.writeUInt8(c.sampleRateDivider, i);
    i = packet.writeUInt16LE(SLEEP_DURATION_S, i);
    i = packet.writeUInt16LE(RECORD_DURATION_S, i);
    i = packet.writeUInt8(LED_ENABLED ? 1 : 0, i);

    i = packet.writeUInt8(RECORDING_PERIODS.length, i);
    for (let p = 0; p < MAX_RECORDING_PERIODS; p++) {
        const [start, end] = RECORDING_PERIODS[p] || [0, 0];
        i = packet.writeUInt16LE(start, i);
        i = packet.writeUInt16LE(end, i);
    }

    const tzHours = Math.trunc(TIMEZONE_OFFSET_MINS / 60);
    i = packet.writeInt8(tzHours, i);
    i = packet.writeUInt8(1, i);
    i = packet.writeUInt8((BATTERY_LEVEL_ENABLED ? 0 : 1) | (PRERECORDING_PREP_TIME << 4), i);
    i = packet.writeInt8(TIMEZONE_OFFSET_MINS - tzHours * 60, i);
    i = packet.writeUInt8(0, i);

    return packet;

}


// Configure using HID

const MAXIMUM_RETRIES = 10;
const RETRY_INTERVAL_MS = 100;

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function hid (fn, ...args) {

    let lastError;

    for (let attempt = 0; attempt < MAXIMUM_RETRIES; attempt++) {
        try {
            return await util.promisify(fn)(...args);
        } catch (err) {
            lastError = err;
            await delay(RETRY_INTERVAL_MS);
        }
    }

    throw new Error('Could not flash AudioMoth (' + lastError + ')');

}


// Download AudioMoth Firmware

const RELEASES_URL = 'https://api.github.com/repos/OpenAcousticDevices/AudioMoth-Firmware-Basic/releases/latest';
const FIRMWARE_DESCRIPTION = 'AudioMoth-Firmware-Basic';

const BOOTLOADER_VENDOR_IDS = ['10C4', '2544'];
const BOOTLOADER_PRODUCT_ID = '0003';

const parseVersion = (text) => (text.match(/\d+/g) || []).slice(0, 3).map(Number);
const versionString = (version) => version.join('.');
const sameVersion = (a, b) => versionString(a) === versionString(b);

function firmwareDirectory () {

    const directory = path.join(app.getPath('userData'), 'firmware');
    fs.mkdirSync(directory, { recursive: true });
    return directory;

}

async function getLatestFirmware () {

    const directory = firmwareDirectory();

    try {

        const response = await net.fetch(RELEASES_URL, { headers: { Accept: 'application/vnd.github+json' } });
        if (!response.ok) throw new Error('GitHub responded ' + response.status);
        const release = await response.json();
        const asset = release.assets.find((a) => a.name.endsWith('.bin'));
        const firmwarePath = path.join(directory, asset.name);

        if (!fs.existsSync(firmwarePath)) {
            const download = await net.fetch(asset.browser_download_url);
            if (!download.ok) throw new Error('Download responded ' + download.status);
            fs.writeFileSync(firmwarePath + '.part', Buffer.from(await download.arrayBuffer()));
            fs.renameSync(firmwarePath + '.part', firmwarePath);
        }

        return { path: firmwarePath, version: parseVersion(release.tag_name) };

    } catch (err) {

        const cached = fs.readdirSync(directory).filter((f) => f.endsWith('.bin'));
        if (cached.length === 0) throw new Error('Could not download the AudioMoth firmware. Check your internet connection and try again.');

        const newest = cached.map((f) => ({ path: path.join(directory, f), version: parseVersion(f) }))
            .sort((a, b) => a.version[0] - b.version[0] || a.version[1] - b.version[1] || a.version[2] - b.version[2])
            .pop();
        return newest;

    }

}

async function findBootloaderPort () {

    const ports = await SerialPort.list();
    const port = ports.find((p) => p.vendorId && p.productId &&
        BOOTLOADER_VENDOR_IDS.includes(p.vendorId.toUpperCase()) && p.productId === BOOTLOADER_PRODUCT_ID);

    return port ? port.path.replace('/dev/cu.', '/dev/tty.') : null;

}

function flashToolPath () {

    const name = process.platform === 'win32' ? 'flash.exe' : 'flash';
    if (app.isPackaged) return path.join(process.resourcesPath, 'bin', name);
    const platformDirectory = { win32: 'windows', darwin: 'macos' }[process.platform] || 'linux';
    return path.join(__dirname, 'bin', platformDirectory, name);

}

// Flash AudioMoth

const FLASH_SETTLE_DELAY_MS = 1000;
const MAX_READY_ATTEMPTS = 7;
const READY_RETRY_BASE_DELAY_MS = 200;
const READY_TIMEOUT_ERROR = 'Timed out waiting for send confirmation from AudioMoth';

async function flashFirmware (port, firmwarePath) {

    await delay(FLASH_SETTLE_DELAY_MS);

    for (let attempt = 1; ; attempt++) {

        const { stdout } = await util.promisify(execFile)(flashToolPath(), ['-u', port, firmwarePath], { timeout: 300000, windowsHide: true });

        const error = stdout.match(/ERROR: (.*)/);
        if (!error) {
            if (!stdout.includes('Programmed:')) throw new Error('Unexpected response from the flash tool: ' + stdout.trim());
            return;
        }

        const message = error[1].trim();
        if (message !== READY_TIMEOUT_ERROR || attempt === MAX_READY_ATTEMPTS) throw new Error(message + ' (attempt ' + attempt + ')');

        await delay(READY_RETRY_BASE_DELAY_MS * Math.pow(2, attempt));

    }

}

const BOOTLOADER_SWITCH_TIMEOUT_MS = 10000;
const RESTART_TIMEOUT_MS = 15000;

async function waitFor (check, timeout) {

    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
        const result = await check().catch(() => null);
        if (result) return result;
        await delay(250);
    }
    return null;

}

async function installFirmware (port, firmwarePath, report) {

    report('Installing firmware. Do not unplug the AudioMoth...');
    await flashFirmware(port, firmwarePath);

    report('Waiting for the AudioMoth to restart...');
    const restarted = await waitFor(() => hid(audiomoth.getFirmwareVersion), RESTART_TIMEOUT_MS);
    if (!restarted) throw new Error('The AudioMoth did not restart after installing firmware. Unplug it, plug it back in, and try again.');

}

async function applySettings (report) {

    report('Configuring settings...');

    let sendTime = Math.floor(Date.now() / 1000) + 1;
    if (sendTime * 1000 - Date.now() < 100) sendTime += 1;
    const packet = buildPacket(sendTime);
    await delay(sendTime * 1000 - Date.now());

    const response = await hid(audiomoth.setPacket, packet);
    if (response === null) throw new Error('The AudioMoth was disconnected while applying settings. Reconnect it and try again.');

    const echoed = Buffer.from(response.slice(1, 1 + PACKET_LENGTH));
    if (!echoed.equals(packet)) throw new Error('The AudioMoth did not configure properly. Unplug it, plug it back in, and try again.');

}

async function flashAudioMoth (report) {

    report('Checking for the latest firmware...');
    const firmware = await getLatestFirmware();

    report('Looking for an AudioMoth...');
    const current = await hid(audiomoth.getFirmwareVersion);

    if (current === null) {

        const port = await findBootloaderPort();
        if (!port) throw new Error('No AudioMoth found. Set the switch to USB/OFF, connect the AudioMoth to this computer with a USB cable, and try again.');
        await installFirmware(port, firmware.path, report);

    } else if (sameVersion(current, firmware.version) && await hid(audiomoth.getFirmwareDescription) === FIRMWARE_DESCRIPTION) {

        report('Firmware ' + versionString(firmware.version) + ' is already installed.');

    } else {

        // For non-HID compatible devices
        const supportsSwitch = await hid(audiomoth.queryBootloader).catch(() => false);
        if (!supportsSwitch) throw new Error('This AudioMoth\'s current firmware cannot be updated automatically. Update it once with the official AudioMoth Flash App, then try again.');

        report('Switching the AudioMoth to flash mode...');
        if (!await hid(audiomoth.switchToBootloader)) throw new Error('The AudioMoth refused to switch to flash mode. Unplug it, plug it back in, and try again.');

        const port = await waitFor(findBootloaderPort, BOOTLOADER_SWITCH_TIMEOUT_MS);
        if (!port) throw new Error('The AudioMoth did not enter flash mode. Unplug it, plug it back in, and try again.');
        await installFirmware(port, firmware.path, report);

    }

    const installed = await hid(audiomoth.getFirmwareVersion);
    if (!installed || !sameVersion(installed, firmware.version)) throw new Error('The firmware could not be verified after installing. Unplug the AudioMoth, plug it back in, and try again.');

    await applySettings(report);

    const deviceID = await hid(audiomoth.getID) || 'unknown';
    return 'AudioMoth ' + deviceID + ' is running firmware ' + versionString(firmware.version) +
        ' with configured settings. You can now unplug it.';

}

// Analyze Data (see analyze.py)

const RESULTS_FOLDER_NAME = 'Soundscape Results';

let analysisProcess = null;

function analyzerCommand () {

    if (app.isPackaged) {
        const executable = path.join(process.resourcesPath, 'analyzer', process.platform === 'win32' ? 'analyzer.exe' : 'analyzer');
        return { command: executable, args: [], models: path.join(process.resourcesPath, 'models') };
    }

    const python = process.platform === 'win32' ? path.join(__dirname, '.venv', 'Scripts', 'python.exe') : path.join(__dirname, '.venv', 'bin', 'python');
    return { command: python, args: [path.join(__dirname, 'analyze.py')], models: path.join(__dirname, 'models') };

}

ipcMain.handle('choose-recordings', async (event) => {

    const result = await dialog.showOpenDialog(BrowserWindow.fromWebContents(event.sender), {
        title: 'Choose the AudioMoth SD card or recordings folder',
        properties: ['openDirectory']
    });

    if (result.canceled) return null;

    return result.filePaths[0];

});

const OUTPUT_SETTING_FILE = () => path.join(app.getPath('userData'), 'save-folder.txt');

function outputFolder () {

    try {
        return fs.readFileSync(OUTPUT_SETTING_FILE(), 'utf8').trim() || path.join(app.getPath('documents'), RESULTS_FOLDER_NAME);
    } catch {
        return path.join(app.getPath('documents'), RESULTS_FOLDER_NAME);
    }

}

ipcMain.handle('output-folder', () => outputFolder());

ipcMain.handle('choose-output', async (event) => {

    const result = await dialog.showOpenDialog(BrowserWindow.fromWebContents(event.sender), {
        title: 'Choose where to save the upload folder',
        defaultPath: outputFolder(),
        properties: ['openDirectory', 'createDirectory']
    });

    if (result.canceled) return null;
    fs.writeFileSync(OUTPUT_SETTING_FILE(), result.filePaths[0]);
    return result.filePaths[0];

});

ipcMain.handle('analyze', (event, { folder, output, location, device }) => {

    const locationSlug = slug(location.name);
    const deviceSlug = slug(device.name);
    if (!locationSlug || !deviceSlug) return { ok: false, message: 'This location or AudioMoth name has no letters or numbers.' };

    if (process.platform === 'darwin' && process.arch === 'x64') {
        return { ok: false, message: 'Analysis needs a Mac with Apple silicon (M1 or newer). Please use a different computer.' };
    }

    if (output === path.join(app.getPath('documents'), RESULTS_FOLDER_NAME)) fs.mkdirSync(output, { recursive: true });
    if (!fs.existsSync(output)) return { ok: false, message: 'The save folder could not be found: ' + output + '. Choose another save folder.' };

    const { command, args, models } = analyzerCommand();

    return new Promise((resolve) => {

        let result = { ok: false, message: 'The analysis stopped unexpectedly.' };
        let stderr = '';
        let pending = '';

        analysisProcess = spawn(command, [...args, '--input', folder, '--output', output, '--location', locationSlug, '--device', deviceSlug,
            '--models', models, '--context', JSON.stringify({
                app: { name: app.getName(), version: app.getVersion(), platform: process.platform, arch: process.arch },
                user, location, device
            })], { windowsHide: true });

        analysisProcess.stdout.on('data', (data) => {
            pending += data.toString();
            const lines = pending.split('\n');
            pending = lines.pop();
            for (const line of lines) {
                let message;
                try { message = JSON.parse(line); } catch { continue; }
                if (message.type === 'progress') event.sender.send('analysis-progress', message);
                if (message.type === 'done') result = { ok: true, path: message.path, message: 'Done! ' + message.analyzed + ' recordings analyzed' + (message.skipped ? ' (' + message.skipped + ' could not be read)' : '') + '. Your upload folder is ready.' };
                if (message.type === 'error') result = { ok: false, message: message.message };
            }
        });

        analysisProcess.stderr.on('data', (data) => { stderr = (stderr + data.toString()).slice(-2000); });

        analysisProcess.on('error', (err) => { result = { ok: false, message: 'Could not start the analysis engine: ' + err.message }; });

        analysisProcess.on('close', (code, signal) => {
            analysisProcess = null;
            if (signal) result = { ok: false, message: 'Analysis cancelled.' };
            else if (!result.ok && code !== 0 && result.message === 'The analysis stopped unexpectedly.') result.message += ' ' + stderr.trim().split('\n').pop();
            resolve(result);
        });

    });

});

ipcMain.handle('cancel-analysis', () => {

    if (analysisProcess) analysisProcess.kill();

});

ipcMain.handle('show-folder', (_event, folder) => shell.showItemInFolder(folder));

// Soundscape Webportal Login

try { process.loadEnvFile(path.join(__dirname, '.env')); } catch {}

const API_URL = 'https://seattle-soundscape.uw.edu/api/v1';

let token = null;
let user = null;

async function portal (route, options = {}) {

    let response;
    try {
        response = await net.fetch(API_URL + route, {
            method: options.body ? 'POST' : (options.method || 'GET'),
            headers: { 'Content-Type': 'application/json', ...(token && { Authorization: 'Bearer ' + token }) },
            body: options.body && JSON.stringify(options.body)
        });
    } catch {
        throw new Error('Could not connect to login service. Check your internet connection and try again.');
    }

    if (response.status === 204) return null;
    const data = await response.json().catch(() => ({}));

    if (response.status === 401 && token) {
        token = null;
        user = null;
        const err = new Error('You have been signed out. Please sign in again.');
        err.signedOut = true;
        throw err;
    }
    if (!response.ok) throw new Error(data.error || 'The Soundscape website returned an error (' + response.status + ').');

    return data;

}

function slug (name) {

    return name.normalize('NFKD').split(/[^A-Za-z0-9-]+/).map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join('');

}

function handlePortal (channel, handler) {

    ipcMain.handle(channel, async (_event, ...args) => {
        try {
            return await handler(...args);
        } catch (err) {
            return { error: err.message, signedOut: Boolean(err.signedOut) };
        }
    });

}

handlePortal('login', async ({ email, password }) => {

    const data = await portal('/auth/login', { body: { email, password, label: app.getName() + ' (' + process.platform + ')' } });
    token = data.token;
    user = data.user;
    return { user };

});

handlePortal('logout', async () => {

    await portal('/auth/logout', { method: 'POST' }).catch(() => {});
    token = null;
    user = null;
    return {};

});

handlePortal('locations', async () => (await portal('/locations')).locations);

handlePortal('audiomoths', async (locationID) => (await portal('/audiomoths?location_id=' + encodeURIComponent(locationID))).audiomoths);

// App Window

let flashing = false;

ipcMain.handle('flash', async (event) => {

    flashing = true;
    try {
        const message = await flashAudioMoth((text) => event.sender.send('flash-progress', text));
        return { ok: true, message };
    } catch (err) {
        return { ok: false, message: err.message || String(err) };
    } finally {
        flashing = false;
    }

});

// Battery Voltage Test

const BATTERY_OK_VOLTS = 4.5;

ipcMain.handle('battery', async () => {

    if (flashing) return null;
    const state = await util.promisify(audiomoth.getBatteryState)().catch(() => null);
    if (!state) return null;
    const volts = parseFloat(state.replace(/[^\d.]/g, ''));
    const ok = state.startsWith('>') || (!state.startsWith('<') && volts >= BATTERY_OK_VOLTS);
    return { state, ok };

});

app.whenReady().then(() => {

    if (process.platform === 'darwin' && !app.isPackaged) app.dock.setIcon(path.join(__dirname, 'assets', 'icon.png'));

    const window = new BrowserWindow({
        width: 780,
        height: 500,
        minWidth: 640,
        minHeight: 420,
        title: 'CROW',
        autoHideMenuBar: true,
        icon: path.join(__dirname, 'assets', 'icon.png'),
        webPreferences: { nodeIntegration: true, contextIsolation: false }
    });

    window.loadFile('index.html');

});

app.on('before-quit', (event) => {

    if (!token) return;
    event.preventDefault();
    const timeout = new Promise((resolve) => setTimeout(resolve, 2000));
    Promise.race([portal('/auth/logout', { method: 'POST' }).catch(() => {}), timeout]).then(() => {
        token = null;
        app.quit();
    });

});

app.on('window-all-closed', () => app.quit());
