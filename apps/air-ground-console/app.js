/* 空地巡检台：真机回放。任务单、机况、风险和发现跟画面用同一套记录。 */
(() => {
  const $ = (id) => document.getElementById(id);

  const OBJECTS = [["凳子", "凳子"], ["椅子", "椅子"], ["凳", "凳子"], ["蓝色桶", "桶"], ["桶", "桶"], ["灭火器", "灭火器"], ["货架", "货架"], ["锥桶", "锥桶"]];
  const COLORS = [["红色", "红"], ["蓝色", "蓝"], ["黄色", "黄"], ["红", "红"], ["蓝", "蓝"], ["黄", "黄"], ["绿", "绿"]];
  const BODY_DIRS = [["左前方", "左前方"], ["右前方", "右前方"], ["前方", "前方"], ["前面", "前方"], ["左侧", "左侧"], ["右侧", "右侧"], ["后方", "后方"]];
  const WORLD_DIRS = [["东侧", "东侧"], ["西侧", "西侧"], ["北侧", "北侧"], ["南侧", "南侧"], ["东", "东"], ["西", "西"], ["北", "北"], ["南", "南"]];

  const SORTIE = {
    "tello-1719": {
      risk: "报警 · 未锁定",
      klass: "warn",
      why: "前方最近约 1.22 m，是整趟记录，不是当前帧。",
      gate: "户外围挡 · 城市内部门禁未通过",
      find: (far) => `未锁定 · 里程计最远 ${far.d.toFixed(1)} m · 真实 ${fmt(far.t)} · 无锁定帧`,
    },
    "tello-170703": {
      risk: "无逐帧净空",
      klass: "warn",
      why: "这段是约 3 m 的蓝桶回放。目标点取的是轨迹尽头。",
      gate: "户外围挡 · 城市内部门禁未通过",
      find: (far) => `没有锁定结论 · 里程计最远 ${far.d.toFixed(1)} m · 真实 ${fmt(far.t)}`,
    },
  };

  function firstHit(text, table) {
    let best = null;
    for (const [key, name] of table) {
      const i = text.indexOf(key);
      if (i < 0) continue;
      if (!best || i < best.i || (i === best.i && key.length > best.len)) best = { i, len: key.length, name };
    }
    return best;
  }

  function parseInstruction(text) {
    const raw = text.trim();
    if (!raw) return null;
    let label = "巡检点";
    for (const [key, name] of OBJECTS) {
      if (raw.includes(key)) { label = name; break; }
    }
    const color = firstHit(raw, COLORS);
    let distance = null;
    let approx = false;
    const dist = raw.match(/(\d+(?:\.\d+)?)\s*(公里|千米|km|米|m)(?![a-z0-9])/i);
    if (dist) {
      distance = parseFloat(dist[1]) * (/公里|千米|km/i.test(dist[2]) ? 1000 : 1);
      approx = /左右|大约|大概|附近|约/.test(raw.slice(Math.max(0, dist.index - 3), dist.index + dist[0].length + 3));
    }
    const body = firstHit(raw, BODY_DIRS);
    const world = firstHit(raw, WORLD_DIRS);
    let dirName = "当前航向";
    if (body && (!world || body.i <= world.i)) dirName = body.name;
    else if (world) dirName = world.name;
    return {
      text: raw,
      label,
      color: color ? color.name : null,
      distance,
      approx,
      dirName,
      reportDog: /机器狗|机械狗|狗/.test(raw),
    };
  }

  function chip(text, kind) {
    const el = document.createElement("span");
    el.className = kind ? `chip ${kind}` : "chip";
    el.textContent = text;
    return el;
  }

  function renderParse(spec) {
    const el = $("parse-out");
    el.replaceChildren();
    if (!spec) {
      el.append(chip("等待任务指令"));
      return;
    }
    el.append(chip(spec.label, "air"));
    if (spec.color) el.append(chip(spec.color));
    el.append(chip(spec.distance == null ? "距离未指定" : `${spec.approx ? "约 " : ""}${spec.distance} m`));
    el.append(chip(spec.dirName));
    el.append(chip(spec.reportDog ? "交给机器狗" : "仅空中", spec.reportDog ? "dog" : "air"));
  }

  const ROSTER = [
    { id: "tello-1719", type: "air", name: "Tello 1719", clipId: "tello-1719" },
    { id: "tello-170703", type: "air", name: "Tello 170703", clipId: "tello-170703" },
    { id: "dog-231931", type: "ground", name: "Lumos 231931", clipId: "dog-231931" },
    { id: "dog-b", type: "ground", name: "地面 B", clipId: null },
    { id: "dog-c", type: "ground", name: "地面 C", clipId: null },
  ];

  const state = {
    pack: null,
    air: null,
    dog: null,
    rate: 1,
    seeking: false,
    mission: null,
    hold: null,
    dogIn: false,
    reportOn: false,
    focus: { air: "tello-1719", ground: "dog-231931" },
    hideAir: false,
    hideDog: false,
    slots: null,
  };
  state.slots = Object.fromEntries(ROSTER.map((robot) => [robot.id, {
    robot,
    segment: "—",
    status: "空闲",
    follow: robot.clipId ? "有录像，还没派" : "无真机录像",
    missionId: null,
    started: false,
  }]));

  const airVideo = $("air-video");
  const dogVideo = $("dog-video");

  function clipById(id) {
    return state.pack.clips.find((c) => c.id === id);
  }

  function fmt(sec) {
    const s = Math.max(0, Math.floor(sec || 0));
    return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  }

  function hdg(yaw) {
    const deg = ((yaw * 180 / Math.PI) % 360 + 360) % 360;
    return String(Math.round(deg)).padStart(3, "0");
  }

  function setText(id, text) {
    const el = $(id);
    if (el && el.textContent !== text) el.textContent = text;
  }

  function indexOf(clip, video) {
    if (!clip || !video) return 0;
    const i = Math.round((video.currentTime || 0) * clip.fps);
    return Math.max(0, Math.min(clip.points.length - 1, i));
  }

  function pointOf(clip, video) {
    return clip.points[indexOf(clip, video)];
  }

  function farOf(clip) {
    if (clip._far) return clip._far;
    let best = { d: 0, t: 0 };
    for (const p of clip.points) {
      const d = Math.hypot(p.x, p.y);
      if (d >= best.d) best = { d, t: p.t };
    }
    clip._far = best;
    return best;
  }

  function goalDist(clip, p) {
    if (!clip || !clip.goal || !p) return null;
    return Math.hypot(p.x - clip.goal.x, p.y - clip.goal.y);
  }

  function syncAbort() {
    const busy = ROSTER.some((robot) => {
      const status = state.slots[robot.id].status;
      return status === "回放中" || status === "已派";
    });
    $("abort").disabled = !busy;
  }

  function setStatus(text) {
    const el = $("mission-status");
    if (el.textContent !== text) el.textContent = text;
  }

  function rolesOf(spec) {
    if (!spec) return { air: "等待指令", dog: "等待指令" };
    if (/三条通道|三个通道|三条|两条通道/.test(spec.text)) {
      return { air: "本句没有空中段", dog: "同型地面席位按通道分，有录像的先走" };
    }
    if (spec.reportDog) {
      return { air: "补高处和视野盲区，锁定后留在附近监视", dog: "沿通道抵近" };
    }
    return { air: "只飞空中搜索", dog: "本句未交给狗" };
  }

  function resultLine() {
    if (!state.mission) return "尚未下发。中间的表是可派席位。";
    if (state.mission.status === "已中止") return "已中止。停住的是正在回放的那几台，排队的段没有轨迹。";
    const count = (status) => ROSTER.filter((robot) => state.slots[robot.id].status === status).length;
    return `回放 ${count("回放中")} 台 · 已派待看 ${count("已派")} 台 · 排队 ${count("排队")} 段`;
  }

  function renderTicket() {
    const spec = state.mission ? state.mission.spec : parseInstruction($("task-input").value);
    if (!spec) {
      setText("ticket-text", "等待任务指令");
      setText("ticket-actors", "空中与地面都还在待命。");
    } else {
      setText("ticket-text", spec.text);
      const roles = rolesOf(spec);
      setText("role-air", roles.air);
      setText("role-dog", roles.dog);
      if (!state.mission) {
        setText("ticket-id", "未下发");
        setText("ticket-actors", "空中与地面都还在待命。");
        setText("ticket-evidence", "证据未挂上");
      }
    }
    if (state.mission && state.air && state.dog) {
      const mine = ROSTER.map((robot) => state.slots[robot.id]).filter((slot) => slot.missionId === state.mission.id);
      setText("ticket-actors", mine.length ? mine.map((slot) => `${slot.robot.name} · ${slot.segment} · ${slot.status}`).join("　") : "这一句还没有派到席位。");
      const files = mine.filter((slot) => slot.robot.clipId && slot.status !== "排队").map((slot) => `${slot.robot.clipId}.mp4`);
      setText("ticket-evidence", files.length ? `证据 ${files.join(" · ")}` : "排队的段没有证据文件");
    }
    setText("ticket-result", resultLine());
    if (state.air) {
      const meta = SORTIE[state.air.id];
      const risk = $("air-risk");
      risk.className = `risk ${meta.klass}`;
      risk.textContent = meta.risk;
      setText("air-risk-why", meta.why);
      setText("air-find", meta.find(farOf(state.air)));
      const dogPlace = state.dog ? state.dog.place : "地面记录未装上";
      setText("gate-line", `门禁 ${meta.gate}。地面 ${dogPlace}。`);
      setText("air-health", `${state.air.short} · 电量无记录 · 链路回放 · 急停无记录 · 定位里程计`);
    }
    if (state.dog) {
      setText("dog-health", `${state.dog.short} · 电量无记录 · 链路回放 · 急停无记录 · 定位里程计`);
      setText("dog-risk-why", "侧向约 0.63 m 仍在走。这是整趟结论，没有逐帧净空。");
    }
  }

  function applyRate() {
    airVideo.playbackRate = state.rate;
    dogVideo.playbackRate = state.rate;
    document.querySelectorAll("[data-rate]").forEach((btn) => {
      btn.setAttribute("aria-pressed", String(Number(btn.dataset.rate) === state.rate));
    });
  }

  function playing() {
    return !airVideo.paused || !dogVideo.paused;
  }

  function setPlaying(on) {
    applyRate();
    for (const video of [airVideo, dogVideo]) {
      if (!on) {
        video.pause();
        continue;
      }
      if (video.ended || (video.duration && video.currentTime > video.duration - 0.05)) video.currentTime = 0;
      video.play().catch(() => {});
    }
    $("play-toggle").textContent = on ? "暂停" : "播放";
  }

  function seek(t) {
    for (const video of [airVideo, dogVideo]) {
      const dur = Number.isFinite(video.duration) ? video.duration : t;
      const next = Math.min(Math.max(0, t), Math.max(0, dur));
      if (Math.abs((video.currentTime || 0) - next) > 0.05) video.currentTime = next;
    }
  }

  function bindClip(video, clip) {
    if (video.getAttribute("src") !== clip.src) {
      video.src = clip.src;
      video.load();
    }
    video.playbackRate = state.rate;
  }

  function paintSite() {
    const site = $("site-line");
    site.replaceChildren(document.createTextNode(`${state.air.when} ${state.air.place}`));
    const sub = document.createElement("span");
    sub.textContent = `${state.dog.when} ${state.dog.place} · 里程计 · 不同场地`;
    site.append(sub);
  }

  function useAir(id) {
    const clip = clipById(id);
    if (!clip) return null;
    state.air = clip;
    bindClip(airVideo, clip);
    $("air-name").textContent = clip.short;
    $("air-when").textContent = clip.when.slice(5);
    paintSite();
    renderTicket();
    return clip;
  }

  function restart(video) {
    video.currentTime = 0;
    video.playbackRate = state.rate;
    video.play().catch(() => {});
    $("play-toggle").textContent = "暂停";
  }

  function missionId() {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    return `T-${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  }

  function tone(status) {
    if (status === "回放中") return "run";
    if (status === "已派") return "hold";
    if (status === "排队" || status === "未执行") return "queue";
    if (status === "已中止") return "stop";
    return "idle";
  }

  function renderFleet() {
    const root = $("fleet-rows");
    if (!root) return;
    if (!root.childElementCount) {
      for (const robot of ROSTER) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `frow ${robot.type}`;
        btn.dataset.robot = robot.id;
        for (const cls of ["who", "seg", "stat", "follow"]) {
          const span = document.createElement("span");
          span.className = cls;
          btn.append(span);
        }
        btn.querySelector(".who").classList.add(robot.type);
        root.append(btn);
      }
    }
    for (const robot of ROSTER) {
      const slot = state.slots[robot.id];
      const btn = root.querySelector(`[data-robot="${robot.id}"]`);
      btn.setAttribute("aria-pressed", String(state.focus[robot.type] === robot.id));
      btn.querySelector(".who").textContent = robot.name;
      btn.querySelector(".seg").textContent = slot.segment;
      const stat = btn.querySelector(".stat");
      stat.textContent = slot.status;
      stat.className = `stat ${tone(slot.status)}`;
      btn.querySelector(".follow").textContent = slot.follow;
    }
  }

  function liveFollow(robot) {
    if (!state.pack || !robot.clipId) return null;
    const showing = robot.type === "air" ? state.air : state.dog;
    if (!showing || showing.id !== robot.clipId) return null;
    const video = robot.type === "air" ? airVideo : dogVideo;
    const point = pointOf(showing, video);
    const dist = goalDist(showing, point);
    if (robot.type === "ground" && dist <= 1) return `进入到达圈 · 积分 ${dist.toFixed(2)} m · 真实 ${fmt(point.t)}`;
    return `距目标 ${dist.toFixed(1)} m · 真实 ${fmt(point.t)}`;
  }

  function freeze(slot) {
    const text = liveFollow(slot.robot);
    if (text) slot.follow = text.replace("距目标", "停在距目标").replace("进入到达圈", "已进圈");
  }

  function showEmpty(robot) {
    const video = robot.type === "air" ? airVideo : dogVideo;
    const empty = $(robot.type === "air" ? "air-empty" : "dog-empty");
    video.classList.add("is-hidden");
    empty.hidden = false;
    empty.textContent = `${robot.name} 没有真机录像。这一段留在队列里，不播画面，也不画轨迹。`;
    if (robot.type === "air") state.hideAir = true;
    else state.hideDog = true;
  }

  function showClip(robot, fromStart) {
    if (!state.pack || !robot.clipId) return;
    const video = robot.type === "air" ? airVideo : dogVideo;
    const empty = $(robot.type === "air" ? "air-empty" : "dog-empty");
    video.classList.remove("is-hidden");
    empty.hidden = true;
    if (robot.type === "air") state.hideAir = false;
    else state.hideDog = false;
    for (const other of ROSTER) {
      if (other.type !== robot.type || other.id === robot.id) continue;
      const slot = state.slots[other.id];
      if (slot.status === "回放中") {
        freeze(slot);
        slot.status = "已派";
        slot.follow = `已派 · ${slot.follow}`;
      }
    }
    if (robot.type === "air") useAir(robot.clipId);
    else {
      state.dog = clipById(robot.clipId);
      bindClip(dogVideo, state.dog);
      $("dog-name").textContent = state.dog.short;
      $("dog-when").textContent = state.dog.when.slice(5);
    }
    if (fromStart) restart(video);
    state.focus[robot.type] = robot.id;
  }

  function focusRobot(id) {
    const robot = ROSTER.find((item) => item.id === id);
    const slot = state.slots[id];
    if (!robot || !slot) return;
    if (slot.status === "排队") {
      state.focus[robot.type] = id;
      showEmpty(robot);
      setStatus(`${robot.name} 分到${slot.segment}。没有录像，所以没有轨迹。`);
      renderFleet();
      return;
    }
    if (!robot.clipId) {
      state.focus[robot.type] = id;
      showEmpty(robot);
      setStatus(`${robot.name} 没有真机录像。`);
      renderFleet();
      return;
    }
    const samePlaying = ROSTER.some((item) => item.type === robot.type && item.id !== id && state.slots[item.id].status === "回放中");
    if ((slot.status === "空闲" || slot.status === "未执行" || slot.status === "已中止") && samePlaying) {
      setStatus(`${robot.name} 还没派到当前这段。同型已有一台在回放，主画面留在那一台。`);
      return;
    }
    if (slot.status === "回放中") {
      showClip(robot, false);
      renderFleet();
      return;
    }
    if (slot.status === "已派") {
      showClip(robot, !slot.started);
      slot.status = "回放中";
      slot.started = true;
      setStatus(`主画面改到 ${robot.name} · ${slot.segment}。`);
      renderFleet();
      return;
    }
    showClip(robot, false);
    setStatus(`在看 ${robot.name} 的录像。这一行还是空闲，没有派任务。`);
    renderFleet();
  }

  function segmentsOf(spec) {
    const text = spec.text;
    const unsupported = /灭火器|货架|锥桶/.test(text);
    const items = [];
    let aisles = 0;
    if (/三条通道|三个通道|三条/.test(text)) aisles = 3;
    else if (/两条通道|两只/.test(text)) aisles = 2;
    if (aisles) {
      for (let i = 1; i <= aisles; i += 1) items.push({ type: "ground", segment: `通道 ${i}` });
    } else if (spec.reportDog || /12\s*米/.test(text)) {
      items.push({ type: "ground", segment: spec.distance ? `抵近 ${spec.distance} m` : "通道抵近" });
    }
    const stool = /凳|椅/.test(text);
    const barrel = /桶/.test(text) && !/锥桶/.test(text);
    if (stool && barrel) {
      items.push({ type: "air", segment: "红凳", clipId: "tello-1719" });
      items.push({ type: "air", segment: "蓝桶", clipId: "tello-170703" });
    } else if (barrel) items.push({ type: "air", segment: "蓝桶", clipId: "tello-170703" });
    else if (stool) items.push({ type: "air", segment: "红凳", clipId: "tello-1719" });
    return { unsupported, items, aisles };
  }

  function pickSlot(type, clipId) {
    const slots = ROSTER.filter((robot) => robot.type === type).map((robot) => state.slots[robot.id]);
    const free = (slot) => slot.status === "空闲" || slot.status === "未执行" || slot.status === "已结束" || slot.status === "已中止";
    if (clipId) {
      const hinted = slots.find((slot) => slot.robot.clipId === clipId && free(slot));
      if (hinted) return hinted;
      const other = slots.find((slot) => slot.robot.clipId && slot.robot.clipId !== clipId && free(slot));
      if (other) return other;
    } else {
      const media = slots.find((slot) => slot.robot.clipId && free(slot));
      if (media) return media;
    }
    return slots.find((slot) => !slot.robot.clipId && free(slot)) || null;
  }

  function claim(slot, segment, missionId) {
    slot.segment = segment;
    slot.missionId = missionId;
    if (!slot.robot.clipId) {
      slot.status = "排队";
      slot.follow = "没有真机录像，不画轨迹";
      slot.started = false;
      return;
    }
    const playing = ROSTER.some((robot) => robot.type === slot.robot.type && state.slots[robot.id].status === "回放中");
    if (playing) {
      slot.status = "已派";
      slot.started = false;
      slot.follow = "已派到这段。点这一行才换到主画面";
      return;
    }
    slot.status = "回放中";
    slot.started = true;
    slot.follow = "刚派下";
    showClip(slot.robot, true);
    slot.status = "回放中";
  }

  function dispatch(spec) {
    const { unsupported, items, aisles } = segmentsOf(spec);
    if (unsupported && !items.length) {
      setStatus("档案里没有灭火器、货架或锥桶的真机录像。没有生成任务单。");
      return;
    }
    if (!items.length) {
      setStatus("这句话没有可派的空中或地面段。");
      return;
    }
    const mission = { id: missionId(), spec, status: "回放中", dogOn: items.some((item) => item.type === "ground"), airOn: items.some((item) => item.type === "air") };
    state.mission = mission;
    setText("ticket-id", mission.id);
    const notes = [];
    if (unsupported) notes.push("档案里没有灭火器、货架或锥桶的真机录像");
    if (aisles) {
      setText("role-air", "本句没有空中段");
      setText("role-dog", "同型地面席位按通道分，有录像的先走");
    }
    for (const item of items) {
      const slot = pickSlot(item.type, item.clipId);
      if (!slot) {
        notes.push(`${item.segment} 没有剩余席位`);
        continue;
      }
      const wasBusy = ROSTER.some((robot) => robot.type === item.type && state.slots[robot.id].status === "回放中");
      claim(slot, item.segment, mission.id);
      if (slot.status === "排队") notes.push(`${slot.robot.name} 接到${item.segment}，无录像，留在队列`);
      else if (slot.status === "已派") notes.push(`${slot.robot.name} 接到${item.segment}。同型已有一台在回放，点这一行再看`);
      else notes.push(`${slot.robot.name} 回放${item.segment}`);
      if (wasBusy && slot.status === "回放中") notes.push("同型空闲席位接了新段");
    }
    notes.push("移交仍要同一坐标系，按钮保持关闭");
    renderTicket();
    renderFleet();
    setStatus(notes.join("。") + "。");
    syncAbort();
    if (state.reportOn) renderReport();
  }

  function abortMission() {
    const active = ROSTER.filter((robot) => {
      const status = state.slots[robot.id].status;
      return status === "回放中" || status === "已派" || status === "排队";
    });
    if (!active.length) return;
    setPlaying(false);
    const airPoint = state.air && !state.hideAir ? pointOf(state.air, airVideo) : null;
    const dogPoint = state.dog && !state.hideDog ? pointOf(state.dog, dogVideo) : null;
    state.hold = { air: airPoint, dog: dogPoint };
    for (const robot of ROSTER) {
      const slot = state.slots[robot.id];
      if (slot.status === "回放中") {
        freeze(slot);
        slot.status = "已中止";
      } else if (slot.status === "已派") {
        slot.status = "未执行";
        slot.follow = "还没换到主画面，中止时没有这段轨迹";
      } else if (slot.status === "排队") {
        slot.status = "未执行";
        slot.follow = "没有录像，中止前也没有轨迹";
      }
    }
    if (state.mission) state.mission.status = "已中止";
    renderTicket();
    renderFleet();
    setStatus("正在回放的席位已停。排队和未换画面的段没有轨迹。飞机没有收到悬停指令，地面也没有收到停车指令。");
    syncAbort();
    if (state.reportOn) renderReport();
  }

  function grab(video, canvas) {
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!video.videoWidth) return;
    const scale = Math.max(canvas.width / video.videoWidth, canvas.height / video.videoHeight);
    const w = video.videoWidth * scale;
    const h = video.videoHeight * scale;
    ctx.drawImage(video, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
  }

  function renderReport() {
    const spec = state.mission ? state.mission.spec : parseInstruction($("task-input").value);
    const id = state.mission ? state.mission.id : "未下发";
    const text = spec ? spec.text : "没有指令";
    setText("report-brief", `${id} · ${text}`);
    const air = state.air ? SORTIE[state.air.id].find(farOf(state.air)) : "空中记录未装上";
    const dog = $("dog-find").textContent;
    const handoff = "移交未发生。两机不是同一坐标系。";
    setText("report-gap", `${air}。${dog}。${handoff} 未完成原因写在发现里：空中没有锁定帧，地面到达是里程计口径。`);
    grab(airVideo, $("air-frame"));
    grab(dogVideo, $("dog-frame"));
  }

  function setReport(on) {
    state.reportOn = on;
    $("findings").hidden = on;
    $("report").hidden = !on;
    $("side-title").textContent = on ? "一页报告" : "发现";
    $("report-toggle").setAttribute("aria-pressed", String(on));
    $("report-toggle").textContent = on ? "返回发现" : "一页报告";
    if (on) renderReport();
  }

  function bindForm() {
    const input = $("task-input");
    const submit = $("task-submit");
    const refresh = () => {
      const spec = parseInstruction(input.value);
      submit.disabled = !spec;
      renderParse(spec);
      if (!state.mission) renderTicket();
      if (state.reportOn) renderReport();
    };
    input.addEventListener("input", refresh);
    $("task-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const spec = parseInstruction(input.value);
      if (spec && state.pack) dispatch(spec);
    });
    refresh();
  }

  function bindClicks() {
    window.addEventListener("click", (event) => {
      const el = event.target instanceof Element ? event.target.closest("button, video") : null;
      if (!el || el.disabled) return;
      if (el.id === "task-submit") {
        event.preventDefault();
        $("task-form").requestSubmit();
        return;
      }
      if (el.id === "abort") abortMission();
      else if (el.dataset.robot) focusRobot(el.dataset.robot);
      else if (el.id === "report-toggle") setReport(!state.reportOn);
      else if (el.id === "frame-refresh") renderReport();
      else if (el.id === "play-toggle" || el === airVideo || el === dogVideo) setPlaying(!playing());
      else if (el.dataset.rate) {
        state.rate = Number(el.dataset.rate);
        applyRate();
      } else if (el.dataset.fill) {
        $("task-input").value = el.dataset.fill;
        $("task-input").dispatchEvent(new Event("input", { bubbles: true }));
        $("task-input").focus();
      }
    }, true);
  }

  function surface(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    const pw = Math.round(w * dpr);
    const ph = Math.round(h * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function mapper(w, h) {
    const pts = [];
    const visible = [];
    if (!state.hideAir && state.air) visible.push(state.air);
    if (!state.hideDog && state.dog) visible.push(state.dog);
    for (const clip of visible) {
      pts.push(...clip.points);
      if (clip.goal) pts.push(clip.goal);
    }
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of pts) {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }
    if (!Number.isFinite(minX)) { minX = -5; maxX = 5; minY = -5; maxY = 5; }
    const pad = 3;
    minX -= pad; maxX += pad; minY -= pad; maxY += pad;
    const spanX = Math.max(8, maxX - minX);
    const spanY = Math.max(8, maxY - minY);
    const s = Math.min((w - 36) / spanX, (h - 36) / spanY);
    const ox = (w - spanX * s) / 2;
    const oy = (h - spanY * s) / 2;
    return {
      s, minX, maxX, minY, maxY,
      p(x, y) { return { X: ox + (x - minX) * s, Y: oy + (maxY - y) * s }; },
    };
  }

  function strokePath(ctx, map, points, to, color, width, dash) {
    if (to < 1) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    for (let i = 0; i <= to; i++) {
      const m = map.p(points[i].x, points[i].y);
      if (i === 0) ctx.moveTo(m.X, m.Y);
      else ctx.lineTo(m.X, m.Y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawMarker(ctx, map, p, color, yaw) {
    const m = map.p(p.x, p.y);
    ctx.save();
    ctx.translate(m.X, m.Y);
    ctx.rotate(yaw - Math.PI / 2);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(8, 0);
    ctx.lineTo(-6, 4.2);
    ctx.lineTo(-3, 0);
    ctx.lineTo(-6, -4.2);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function drawHold(ctx, map) {
    if (!state.hold) return;
    for (const [p, color] of [[state.hold.air, "#7ec8c0"], [state.hold.dog, "#e2a24a"]]) {
      const m = map.p(p.x, p.y);
      ctx.save();
      ctx.strokeStyle = "#d36a5a";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(m.X, m.Y, 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(m.X, m.Y, 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  function drawMap() {
    const canvas = $("map-view");
    const { ctx, w, h } = surface(canvas);
    const map = mapper(w, h);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#141a12";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(231,220,196,0.08)";
    ctx.lineWidth = 1;
    ctx.font = '11px "PingFang SC", sans-serif';
    ctx.fillStyle = "rgba(231,220,196,0.35)";
    const step = 5;
    const x0 = Math.floor(map.minX / step) * step;
    for (let x = x0; x <= map.maxX + step; x += step) {
      const a = map.p(x, map.maxY);
      if (a.X < -20 || a.X > w + 20) continue;
      ctx.beginPath();
      ctx.moveTo(a.X, 0);
      ctx.lineTo(a.X, h);
      ctx.stroke();
      if (a.X > 8 && a.X < w - 24) ctx.fillText(String(x), a.X + 2, 14);
    }
    for (let y = Math.ceil(map.maxY / step) * step; y >= map.minY - step; y -= step) {
      const a = map.p(map.minX, y);
      if (a.Y < -20 || a.Y > h + 20) continue;
      ctx.beginPath();
      ctx.moveTo(0, a.Y);
      ctx.lineTo(w, a.Y);
      ctx.stroke();
    }

    const drawClip = (clip, video, color, dim) => {
      if (!clip) return;
      const i = indexOf(clip, video);
      strokePath(ctx, map, clip.points, clip.points.length - 1, dim, 1.2, [4, 5]);
      strokePath(ctx, map, clip.points, i, color, 2.4);
      const p = clip.points[i];
      if (clip.goal) {
        const g = map.p(clip.goal.x, clip.goal.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.arc(g.X, g.Y, Math.max(6, map.s * 1), 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(clip.goal.label, g.X + 8, g.Y - 6);
      }
      drawMarker(ctx, map, p, color, p.yaw);
      ctx.fillStyle = color;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const m = map.p(p.x, p.y);
      ctx.fillText(clip.vehicle === "ground" ? "地面" : "空中", m.X + 10, m.Y + (clip.vehicle === "ground" ? 14 : -12));
    };
    if (!state.hideDog) drawClip(state.dog, dogVideo, "#e2a24a", "rgba(226,162,74,0.35)");
    if (!state.hideAir) drawClip(state.air, airVideo, "#7ec8c0", "rgba(126,200,192,0.35)");
    drawHold(ctx, map);

    const o = map.p(5, 0);
    const z = map.p(0, 0);
    ctx.strokeStyle = "rgba(231,220,196,0.75)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(z.X, h - 16);
    ctx.lineTo(z.X + (o.X - z.X), h - 16);
    ctx.stroke();
    ctx.fillStyle = "rgba(231,220,196,0.75)";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("5 m", z.X, h - 20);
  }

  function paint() {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    $("clock").textContent = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
    if (!state.air || !state.dog) return;
    const air = pointOf(state.air, airVideo);
    const dog = pointOf(state.dog, dogVideo);
    const airMode = airVideo.paused ? (airVideo.ended ? "结束" : "暂停") : "回放";
    const dogMode = dogVideo.paused ? (dogVideo.ended ? "结束" : "暂停") : "回放";
    $("air-mode").textContent = airMode;
    $("dog-mode").textContent = dogMode;
    $("air-mode").className = airMode === "回放" ? "hot" : "";
    $("dog-mode").className = dogMode === "回放" ? "hot" : "";
    const airD = goalDist(state.air, air);
    const dogD = goalDist(state.dog, dog);
    const alt = (v) => Math.max(0, v).toFixed(1);
    $("air-telem").textContent = `高度 ${alt(air.alt)} m    航向 ${hdg(air.yaw)}°    真实 ${fmt(air.t)}    目标 ${airD.toFixed(1)} m`;
    $("dog-telem").textContent = `高度 ${alt(dog.alt)} m    航向 ${hdg(dog.yaw)}°    真实 ${fmt(dog.t)}    目标 ${dogD.toFixed(1)} m`;
    $("air-pose").textContent = `空中  真实 ${fmt(air.t)} · x ${air.x.toFixed(1)}  y ${air.y.toFixed(1)} · 高 ${alt(air.alt)} m`;
    $("dog-pose").textContent = `地面  真实 ${fmt(dog.t)} · x ${dog.x.toFixed(1)}  y ${dog.y.toFixed(1)} · 高 ${alt(dog.alt)} m`;
    setText("air-dist", `空中距目标 ${airD.toFixed(1)} m`);
    setText("dog-dist", `地面距目标 ${dogD.toFixed(1)} m`);
    state.dogIn = dogD <= 1;
    setText(
      "dog-find",
      dogD <= 1
        ? `进入到达圈 · 积分 ${dogD.toFixed(2)} m · 真实 ${fmt(dog.t)} · x ${dog.x.toFixed(1)} y ${dog.y.toFixed(1)}`
        : `距目标 ${dogD.toFixed(1)} m · 真实 ${fmt(dog.t)} · 尚未进入 1 m 圈`,
    );
    if (state.mission && state.mission.status === "回放中") {
      const still = ROSTER.some((robot) => state.slots[robot.id].status === "回放中" || state.slots[robot.id].status === "已派");
      if (!still) state.mission.status = "已结束";
    }
    for (const robot of ROSTER) {
      const slot = state.slots[robot.id];
      const text = liveFollow(robot);
      if (!text) continue;
      if (slot.status === "回放中") slot.follow = text;
      else if (slot.status === "空闲" && state.focus[robot.type] === robot.id) slot.follow = `预览 · ${text}`;
      const video = robot.type === "air" ? airVideo : dogVideo;
      const showing = robot.type === "air" ? state.air : state.dog;
      if (slot.status === "回放中" && showing && showing.id === robot.clipId && video.ended) {
        slot.status = "已结束";
        slot.follow = text;
      }
    }
    renderFleet();
    setText("ticket-result", resultLine());
    syncAbort();
    const master = Math.max(airVideo.currentTime || 0, dogVideo.currentTime || 0);
    const dur = Math.max(airVideo.duration || 0, dogVideo.duration || 0, 1);
    if (!state.seeking) $("scrub").value = String(Math.round(1000 * master / dur));
    $("time-read").textContent = `${fmt(master)} / ${fmt(dur)}`;
    $("play-toggle").textContent = playing() ? "暂停" : "播放";
    drawMap();
  }

  function frame() {
    try { paint(); } catch (err) { setStatus(`画面错误：${err.message}`); }
    requestAnimationFrame(frame);
  }

  function bindTransport() {
    const scrub = $("scrub");
    scrub.addEventListener("pointerdown", () => { state.seeking = true; });
    scrub.addEventListener("input", () => {
      const dur = Math.max(airVideo.duration || 0, dogVideo.duration || 0);
      seek(dur * Number(scrub.value) / 1000);
    });
    scrub.addEventListener("change", () => { state.seeking = false; });
    window.addEventListener("keydown", (event) => {
      if (event.code !== "Space") return;
      if (event.target.closest("input, button, textarea")) return;
      event.preventDefault();
      setPlaying(!playing());
    });
  }

  async function main() {
    bindForm();
    bindClicks();
    bindTransport();
    renderFleet();
    requestAnimationFrame(frame);
    const res = await fetch("media/tracks.json");
    if (!res.ok) throw new Error(`轨迹文件 ${res.status}`);
    state.pack = await res.json();
    state.dog = clipById("dog-231931");
    useAir("tello-1719");
    bindClip(dogVideo, state.dog);
    $("dog-name").textContent = state.dog.short;
    $("dog-when").textContent = state.dog.when.slice(5);
    renderTicket();
    renderFleet();
    setStatus("同型席位在中间这张表。三条通道把有录像的狗派去通道 1，另外两段排队。某一台在回放时，同型的空闲席位还能接新段。");
    setPlaying(true);
  }

  main().catch((err) => setStatus(`真机录像没有装上：${err.message}。请在本目录用本地服务器打开页面。`));
})();
