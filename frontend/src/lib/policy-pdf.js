// PDF renderer for the Educational Policy Brief format.
//
// Input:  { paper, report } where `paper` is the structured prose returned
//         by POST /swarms/policy-report (educational policymaking template),
//         and `report` is the original ResonanceReport JSON.
// Output: triggers a browser download of vishwamitra-policy-brief-<ts>.pdf.
//
// Layout (matches the user's reference template):
//   - Title page: large title, generated date
//   - "What is Educational Policy?" — paragraph
//   - "The Policymaking Process — Key Stages" — header, then six stages
//     each with: bold subheading, lead-in description, bulleted list
//   - "The Iterative Nature of the Process" — bullets
//   - "Key Stakeholders" — bordered table with two columns
//   - "Challenges in Educational Policymaking" — bullets
//   - "Strategies for Effective Implementation" — bullets
//   - "Key Takeaway" — paragraph(s)

import jsPDF from 'jspdf'

// Old export name kept so VerdictPanel.jsx doesn't need to change its import.
export function renderIEEEPolicyPDF({ paper, report }) {
  return renderPolicyBriefPDF({ paper, report })
}

export function renderPolicyBriefPDF({ paper, report }) {
  const pdf = new jsPDF({ unit: 'pt', format: 'letter' })
  const PW = pdf.internal.pageSize.getWidth()
  const PH = pdf.internal.pageSize.getHeight()
  const M = 64
  const W = PW - 2 * M

  let y = M
  let pageNum = 1
  const runningTitle = (paper.title || 'Educational Policy Brief').slice(0, 90)

  // ---------------- helpers ----------------
  function drawFooter() {
    pdf.setFont('times', 'italic')
    pdf.setFontSize(8.5)
    pdf.setTextColor(110)
    pdf.text(runningTitle, M, PH - 28)
    pdf.text(`${pageNum}`, PW - M, PH - 28, { align: 'right' })
    pdf.setLineWidth(0.4)
    pdf.setDrawColor(180)
    pdf.line(M, PH - 36, PW - M, PH - 36)
  }
  function newPage() {
    drawFooter()
    pdf.addPage()
    pageNum += 1
    y = M
  }
  function ensureSpace(needed) {
    if (y + needed > PH - M - 36) newPage()
  }
  function hrThin(weight = 0.4, color = 200) {
    pdf.setLineWidth(weight)
    pdf.setDrawColor(color)
    pdf.line(M, y, PW - M, y)
    y += 10
  }
  function h1Centered(text, size = 22) {
    pdf.setFont('times', 'bold')
    pdf.setFontSize(size)
    pdf.setTextColor(15, 23, 42)
    const lines = pdf.splitTextToSize(text, W)
    for (const line of lines) {
      pdf.text(line, PW / 2, y, { align: 'center' })
      y += size + 4
    }
  }
  function h2(text) {
    ensureSpace(40)
    y += 4
    pdf.setFont('times', 'bold')
    pdf.setFontSize(15)
    pdf.setTextColor(15, 23, 42)
    pdf.text(text, M, y)
    y += 10
    hrThin(0.6, 80)
    y += 4
  }
  function h3(text) {
    ensureSpace(28)
    pdf.setFont('times', 'bold')
    pdf.setFontSize(12.5)
    pdf.setTextColor(20, 30, 50)
    pdf.text(text, M, y)
    y += 16
  }
  function paragraph(text, opts = {}) {
    if (!text || !text.trim()) return
    pdf.setFont('times', opts.italic ? 'italic' : 'normal')
    pdf.setFontSize(opts.size || 10.5)
    pdf.setTextColor(...(opts.color || [30, 35, 50]))
    const paras = text.split(/\n\s*\n+/).map((p) => p.trim().replace(/\s+/g, ' ')).filter(Boolean)
    const lh = (opts.size || 10.5) * 1.42
    for (const para of paras) {
      const lines = pdf.splitTextToSize(para, W - (opts.indent || 0))
      ensureSpace(lines.length * lh + 4)
      lines.forEach((ln) => {
        pdf.text(ln, M + (opts.indent || 0), y)
        y += lh
      })
      y += 4
    }
  }
  function bullets(items, { indent = 14, size = 10.5 } = {}) {
    if (!items || !items.length) return
    pdf.setFontSize(size)
    pdf.setTextColor(30, 35, 50)
    const lh = size * 1.42
    const textWidth = W - indent
    for (const raw of items) {
      const item = String(raw || '').trim().replace(/\s+/g, ' ')
      if (!item) continue
      pdf.setFont('times', 'normal')
      const lines = pdf.splitTextToSize(item, textWidth)
      ensureSpace(lines.length * lh + 4)
      // Bullet glyph
      pdf.setFont('times', 'bold')
      pdf.text('•', M, y)
      pdf.setFont('times', 'normal')
      lines.forEach((ln, i) => {
        pdf.text(ln, M + indent, y)
        y += lh
      })
      y += 2
    }
    y += 2
  }
  function inlineMeta(label, value) {
    if (!value || !value.trim()) return
    ensureSpace(22)
    pdf.setFont('times', 'italic')
    pdf.setFontSize(10)
    pdf.setTextColor(70, 80, 100)
    const labelStr = label + ': '
    pdf.text(labelStr, M + 10, y)
    const offset = pdf.getTextWidth(labelStr)
    pdf.setFont('times', 'normal')
    pdf.setTextColor(40, 45, 60)
    const lines = pdf.splitTextToSize(value.trim(), W - 10 - offset)
    pdf.text(lines[0] || '', M + 10 + offset, y)
    y += 14
    for (let i = 1; i < lines.length; i++) {
      pdf.text(lines[i], M + 10 + offset, y)
      y += 14
    }
    y += 4
  }
  function table(headers, rows) {
    if (!rows || !rows.length) return
    const cellPad = 8
    const col1W = Math.max(140, W * 0.32)
    const col2W = W - col1W
    pdf.setFontSize(10)
    // measure row heights
    const rowHeights = rows.map(([a, b]) => {
      pdf.setFont('times', 'bold')
      const aLines = pdf.splitTextToSize(a || '', col1W - 2 * cellPad)
      pdf.setFont('times', 'normal')
      const bLines = pdf.splitTextToSize(b || '', col2W - 2 * cellPad)
      return Math.max(aLines.length, bLines.length) * 13 + 2 * cellPad
    })
    const headerH = 24
    const total = headerH + rowHeights.reduce((a, b) => a + b, 0)
    ensureSpace(total + 6)

    // header
    pdf.setFillColor(245, 245, 248)
    pdf.rect(M, y, W, headerH, 'F')
    pdf.setLineWidth(0.4)
    pdf.setDrawColor(160)
    pdf.rect(M, y, W, headerH, 'S')
    pdf.line(M + col1W, y, M + col1W, y + headerH)
    pdf.setFont('times', 'bold')
    pdf.setFontSize(10)
    pdf.setTextColor(15, 23, 42)
    pdf.text(headers[0], M + cellPad, y + 16)
    pdf.text(headers[1], M + col1W + cellPad, y + 16)
    let yy = y + headerH

    // rows
    rows.forEach(([a, b], i) => {
      const h = rowHeights[i]
      // alt fill
      if (i % 2 === 1) {
        pdf.setFillColor(252, 252, 254)
        pdf.rect(M, yy, W, h, 'F')
      }
      pdf.setLineWidth(0.4)
      pdf.setDrawColor(160)
      pdf.rect(M, yy, W, h, 'S')
      pdf.line(M + col1W, yy, M + col1W, yy + h)

      pdf.setFont('times', 'bold')
      pdf.setFontSize(10)
      pdf.setTextColor(20, 30, 50)
      const aLines = pdf.splitTextToSize(a || '', col1W - 2 * cellPad)
      aLines.forEach((ln, k) => {
        pdf.text(ln, M + cellPad, yy + cellPad + 10 + k * 13)
      })
      pdf.setFont('times', 'normal')
      pdf.setTextColor(40, 45, 60)
      const bLines = pdf.splitTextToSize(b || '', col2W - 2 * cellPad)
      bLines.forEach((ln, k) => {
        pdf.text(ln, M + col1W + cellPad, yy + cellPad + 10 + k * 13)
      })
      yy += h
    })
    y = yy + 6
  }

  // ---------------- TITLE PAGE ----------------
  pdf.setFillColor(7, 11, 24)
  pdf.rect(0, 0, PW, 4, 'F')

  y = 96
  pdf.setFont('helvetica', 'bold')
  pdf.setFontSize(8)
  pdf.setTextColor(94, 234, 212)
  pdf.text('VISHWAMITRA · POLICY DELIBERATION', PW / 2, y, { align: 'center' })
  y += 36

  h1Centered(paper.title || 'Educational Policy Brief')
  y += 8
  pdf.setFont('times', 'italic')
  pdf.setFontSize(10)
  pdf.setTextColor(80, 90, 110)
  pdf.text(
    `Generated ${new Date(report.timestamp).toLocaleString()}`,
    PW / 2, y, { align: 'center' },
  )
  y += 24
  pdf.setLineWidth(0.6)
  pdf.setDrawColor(120)
  pdf.line(M + 80, y, PW - M - 80, y)
  y += 22

  // ---------------- WHAT IS ----------------
  if (paper.what_is && paper.what_is.trim()) {
    h2('What is Educational Policy?')
    paragraph(paper.what_is)
    y += 6
  }

  // ---------------- KEY STAGES ----------------
  h2('The Policymaking Process — Key Stages')
  y += 4

  const stageDefs = [
    { num: 1, name: 'Issue / Problem Identification',
      desc: paper.stage_1_description, bullets: paper.stage_1_bullets },
    { num: 2, name: 'Agenda Setting',
      desc: paper.stage_2_description, bullets: paper.stage_2_bullets,
      meta: [['Key influencers in this case', paper.stage_2_influencers]] },
    { num: 3, name: 'Policy Formulation',
      desc: paper.stage_3_description, bullets: paper.stage_3_bullets,
      meta: [['Contributors', paper.stage_3_contributors]] },
    { num: 4, name: 'Policy Adoption',
      desc: paper.stage_4_description, bullets: paper.stage_4_bullets },
    { num: 5, name: 'Policy Implementation',
      desc: paper.stage_5_description, bullets: paper.stage_5_bullets,
      meta: [['Primary implementation challenges', paper.stage_5_challenges]] },
    { num: 6, name: 'Policy Evaluation',
      desc: paper.stage_6_description, bullets: paper.stage_6_bullets },
  ]
  for (const s of stageDefs) {
    if (!s.desc && !(s.bullets || []).length) continue
    h3(`Stage ${s.num}: ${s.name}`)
    if (s.desc) paragraph(s.desc)
    bullets(s.bullets)
    if (s.meta) {
      for (const [label, value] of s.meta) inlineMeta(label, value)
    }
    y += 4
  }

  // ---------------- ITERATIVE NATURE ----------------
  if ((paper.iterative_nature || []).length) {
    h2('The Iterative Nature of the Process')
    bullets(paper.iterative_nature)
  }

  // ---------------- STAKEHOLDERS ----------------
  if ((paper.stakeholders || []).length) {
    h2('Key Stakeholders')
    const rows = paper.stakeholders.map((s) => [s.name || '', s.role || ''])
    table(['Stakeholder', 'Role in this deliberation'], rows)
  }

  // ---------------- CHALLENGES ----------------
  if ((paper.challenges || []).length) {
    h2('Challenges in Educational Policymaking')
    bullets(paper.challenges)
  }

  // ---------------- STRATEGIES ----------------
  if ((paper.strategies || []).length) {
    h2('Strategies for Effective Implementation')
    bullets(paper.strategies)
  }

  // ---------------- TAKEAWAY ----------------
  if (paper.takeaway && paper.takeaway.trim()) {
    h2('Key Takeaway')
    paragraph(paper.takeaway)
  }

  // ---------------- APPENDIX: source data ----------------
  ensureSpace(80)
  h2('Appendix · Source Deliberation Snapshot')
  pdf.setFont('times', 'italic')
  pdf.setFontSize(9.5)
  pdf.setTextColor(80, 90, 110)
  pdf.text(
    'The brief above synthesises the swarm-of-swarms deliberation summarised below. '
    + 'Numerical values reproduced verbatim from the deliberation report.',
    M, y,
  )
  y += 16

  const ACTION_NAMES = (report.action_names && report.action_names.length)
    ? report.action_names
    : ['funding_boost','teacher_incentive','student_scholarship','attendance_mandate',
       'resource_realloc','transparency_report','staff_hiring','counseling_programs']
  const final = report.final_action || []
  const reson = report.resonance_per_intervention || []
  const flagSet = new Set(report.dissonance_flags || [])

  // Compact bar table: intervention | intensity | resonance
  const rowH = 16
  const headH = 20
  const colW = [W * 0.30, W * 0.35, W * 0.35]
  ensureSpace(headH + ACTION_NAMES.length * rowH + 8)
  pdf.setFillColor(245, 245, 248)
  pdf.rect(M, y, W, headH, 'F')
  pdf.setLineWidth(0.4)
  pdf.setDrawColor(160)
  pdf.rect(M, y, W, headH, 'S')
  pdf.setFont('times', 'bold')
  pdf.setFontSize(9.5)
  pdf.setTextColor(15, 23, 42)
  pdf.text('Intervention', M + 8, y + 14)
  pdf.text('Recommended intensity', M + colW[0] + 8, y + 14)
  pdf.text('Cross-swarm resonance', M + colW[0] + colW[1] + 8, y + 14)
  let yy = y + headH

  ACTION_NAMES.forEach((n, i) => {
    pdf.setLineWidth(0.3)
    pdf.setDrawColor(200)
    pdf.line(M, yy, M + W, yy)
    pdf.line(M + colW[0], yy, M + colW[0], yy + rowH)
    pdf.line(M + colW[0] + colW[1], yy, M + colW[0] + colW[1], yy + rowH)

    const isFlag = flagSet.has(n)
    pdf.setFont('times', isFlag ? 'bold' : 'normal')
    pdf.setFontSize(9.5)
    pdf.setTextColor(isFlag ? 180 : 40, isFlag ? 30 : 45, isFlag ? 30 : 60)
    pdf.text(n, M + 8, yy + 12)

    // intensity bar
    const f = Math.max(0, Math.min(1, final[i] ?? 0))
    pdf.setFillColor(228, 232, 240)
    pdf.rect(M + colW[0] + 8, yy + 4, colW[1] - 60, 7, 'F')
    pdf.setFillColor(40, 80, 160)
    pdf.rect(M + colW[0] + 8, yy + 4, (colW[1] - 60) * f, 7, 'F')
    pdf.setFont('times', 'normal')
    pdf.setTextColor(40)
    pdf.text(f.toFixed(2), M + colW[0] + colW[1] - 6, yy + 12, { align: 'right' })

    // resonance bar
    const r = Math.max(0, Math.min(1, reson[i] ?? 0))
    pdf.setFillColor(228, 232, 240)
    pdf.rect(M + colW[0] + colW[1] + 8, yy + 4, colW[2] - 60, 7, 'F')
    if (r > 0.75)      pdf.setFillColor(40, 140, 70)
    else if (r > 0.55) pdf.setFillColor(200, 140, 30)
    else               pdf.setFillColor(190, 50, 50)
    pdf.rect(M + colW[0] + colW[1] + 8, yy + 4, (colW[2] - 60) * r, 7, 'F')
    pdf.setTextColor(40)
    pdf.text(r.toFixed(2), M + W - 6, yy + 12, { align: 'right' })

    yy += rowH
  })
  pdf.line(M, yy, M + W, yy)
  pdf.line(M, y, M, yy)
  pdf.line(M + W, y, M + W, yy)
  y = yy + 12

  // dissonance flag list
  if (flagSet.size > 0) {
    pdf.setFont('times', 'italic')
    pdf.setFontSize(9.5)
    pdf.setTextColor(180, 30, 30)
    pdf.text(
      `Dissonance flags: ${[...flagSet].join(', ')}`,
      M, y,
    )
    y += 14
  }

  drawFooter()

  const ts = new Date(report.timestamp).toISOString().replace(/[:.]/g, '-').slice(0, 19)
  pdf.save(`vishwamitra-policy-brief-${ts}.pdf`)
}
