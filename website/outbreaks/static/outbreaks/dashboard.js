const fmt = new Intl.NumberFormat();
const $ = (id) => document.getElementById(id);

const COLORS = {
  green: '#1E4D2B',
  greenDark: '#0b2415',
  greenSoft: '#e8efe9',
  gold: '#C8C372',
  goldBright: '#F1E8A6',
  cream: '#f7f4df',
  muted: '#5f6f5d',
  rust: '#b85635'
};

let map = L.map('map').setView([37.8, -96], 4);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

let detectionLayer = L.layerGroup().addTo(map);
let airportLayer = L.layerGroup().addTo(map);
let countryChart, timeChart, monthlyRiskChart;
let selectedCountry = null;

function metric() { return $('metric').value; }
function state() { return $('state').value; }
function lag() { return $('lag').value; }
function season() { return $('season').value; }
function year() { return $('year').value; }

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

function setSummary(data) {
  const cards = [
    ['Detection rows', data.detections.rows],
    ['Total detections', Math.round(data.detections.total || 0)],
    ['Inbound passengers', Math.round(data.air_traffic.passengers || 0)],
    ['Air freight in lbs', Math.round(data.air_traffic.freight || 0)],
  ];

  $('summaryCards').innerHTML = cards
    .map(([label, value]) => `
      <div class="card">
        <div class="value">${fmt.format(value || 0)}</div>
        <div class="label">${label}</div>
      </div>
    `)
    .join('');
}

function upsertOptions(select, values, firstLabel, firstValue = 'all') {
  select.innerHTML =
    `<option value="${firstValue}">${firstLabel}</option>` +
    values.map(v => `<option value="${v}">${v}</option>`).join('');
}

function renderCountries(payload) {
  const rows = payload.countries || [];
  const distanceLimit = payload.distance_limit_miles ?? 60;
  const top = rows.slice(0, 12);
  const labels = top.map(r => r.country);
  const risk = top.map(r => r.risk_score);

  if (countryChart) countryChart.destroy();

  countryChart = new Chart($('countryChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Risk score',
          data: risk,
          backgroundColor: COLORS.gold,
          borderColor: COLORS.green,
          hoverBackgroundColor: COLORS.goldBright,
          hoverBorderColor: COLORS.greenDark,
          borderWidth: 1.5
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          labels: {
            color: COLORS.greenDark
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: COLORS.muted
          },
          grid: {
            color: 'rgba(30, 77, 43, 0.08)'
          }
        },
        y: {
          ticks: {
            color: COLORS.muted
          },
          grid: {
            color: 'rgba(30, 77, 43, 0.08)'
          }
        }
      }
    }
  });

  $('countryTable').querySelector('tbody').innerHTML = rows.map(r => `
    <tr data-country="${r.country}">
      <td>${r.country}</td>
      <td>${r.correlation === null ? 'n/a' : r.correlation.toFixed(3)}</td>
      <td>${fmt.format(Math.round(r.traffic_total || 0))}</td>
      <td>${fmt.format(r.matched_counties || 0)}</td>
      <td>${r.risk_score.toFixed(2)}</td>
    </tr>
  `).join('');

  document.querySelectorAll('#countryTable tbody tr').forEach(tr => {
    tr.addEventListener('click', () => {
      selectedCountry = tr.dataset.country;
      loadTimeseries();
      loadHotspots();
    });
  });

  if (!selectedCountry && rows.length) {
    selectedCountry = rows[0].country;
  }

  const panelDesc = document.querySelector('.grid.two .panel .panel-title p');
  if (panelDesc) {
    panelDesc.textContent = `Only inbound airport records within ${distanceLimit} miles of a county centroid are included. Risk score combines positive lagged correlation, traffic volume, and detection volume. A 2.5 country of origin scalar is applied if the flies are native to the country importing.`;
  }
}

async function loadCountries() {
  const data = await getJSON(`/api/countries/?metric=${metric()}&state=${encodeURIComponent(state())}&lag=${lag()}`);
  renderCountries(data);
  await loadTimeseries();
}

async function loadTimeseries() {
  if (!selectedCountry) return;

  const data = await getJSON(`/api/timeseries/?state=${encodeURIComponent(state())}&country=${encodeURIComponent(selectedCountry)}`);

  if (timeChart) timeChart.destroy();

  timeChart = new Chart($('timeChart'), {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: [
        {
          label: `${selectedCountry} inbound flights`,
          data: data.traffic,
          yAxisID: 'yTraffic',
          tension: 0.25,
          borderColor: COLORS.gold,
          backgroundColor: COLORS.gold,
          pointBackgroundColor: COLORS.goldBright,
          pointBorderColor: COLORS.green,
          pointRadius: 3,
          pointHoverRadius: 5
        },
        {
          label: 'Fruit fly detections',
          data: data.detections,
          yAxisID: 'yDetect',
          tension: 0.25,
          borderColor: COLORS.rust,
          backgroundColor: COLORS.rust,
          pointBackgroundColor: COLORS.rust,
          pointBorderColor: COLORS.greenDark,
          pointRadius: 3,
          pointHoverRadius: 5
        }
      ]
    },
    options: {
      responsive: true,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          labels: {
            color: COLORS.greenDark
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: COLORS.muted,
            maxTicksLimit: 10
          },
          grid: {
            color: 'rgba(30, 77, 43, 0.08)'
          }
        },
        yTraffic: {
          position: 'left',
          ticks: {
            color: COLORS.muted
          },
          grid: {
            color: 'rgba(30, 77, 43, 0.08)'
          }
        },
        yDetect: {
          position: 'right',
          grid: {
            drawOnChartArea: false
          },
          ticks: {
            color: COLORS.muted
          }
        }
      }
    }
  });
}

async function loadMonthlyRisk() {
  const data = await getJSON(`/api/monthly-risk/?state=${encodeURIComponent(state())}&year=${encodeURIComponent(year())}`);

  if (monthlyRiskChart) monthlyRiskChart.destroy();

  monthlyRiskChart = new Chart($('monthlyRiskChart'), {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [
        {
          label: 'Relative outbreak risk (0-100)',
          data: data.relative_risk,
          backgroundColor: COLORS.rust,
          borderColor: COLORS.greenDark,
          borderWidth: 1.2
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          labels: {
            color: COLORS.greenDark
          }
        },
        tooltip: {
          callbacks: {
            afterLabel: (ctx) => {
              const raw = (data.counts?.[ctx.dataIndex] || 0).toFixed(1);
              return `Detections: ${raw}`;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: COLORS.muted
          },
          grid: {
            color: 'rgba(30, 77, 43, 0.08)'
          }
        },
        y: {
          beginAtZero: true,
          max: 100,
          ticks: {
            color: COLORS.muted
          },
          grid: {
            color: 'rgba(30, 77, 43, 0.08)'
          }
        }
      }
    }
  });
}

function radius(value) {
  return Math.max(4, Math.min(32, Math.sqrt(value || 0) / 40));
}

async function loadHotspots() {
  detectionLayer.clearLayers();
  airportLayer.clearLayers();

  const c = selectedCountry ? `&country=${encodeURIComponent(selectedCountry)}` : '';
  const data = await getJSON(`/api/hotspots/?metric=${metric()}&state=${encodeURIComponent(state())}&season=${season()}&year=${year()}${c}`);

  data.detections.forEach(p => {
    L.circleMarker([p.lat, p.lon], {
      radius: radius(p.value) + 2,
      color: COLORS.rust,
      fillColor: COLORS.rust,
      fillOpacity: 0.45,
      weight: 1
    })
      .bindPopup(`<b>Detection</b><br>${p.name}, ${p.state}<br>${p.year}-${String(p.month).padStart(2, '0')}<br>Count: ${fmt.format(Math.round(p.value))}`)
      .addTo(detectionLayer);
  });

  data.airports.forEach(p => {
    L.circleMarker([p.lat, p.lon], {
      radius: radius(p.value),
      color: COLORS.green,
      fillColor: COLORS.gold,
      fillOpacity: 0.35,
      weight: 1
    })
      .bindPopup(
        `<b>Inbound airport</b><br>${p.name}<br>${p.state}<br>${p.year}-${String(p.month).padStart(2, '0')}<br>${data.metric}: ${fmt.format(Math.round(p.value))}<br>Nearest county: ${p.nearest_county_name}, ${p.nearest_county_state}<br>Distance: ${(p.nearest_county_distance_miles || 0).toFixed(1)} mi`
      )
      .addTo(airportLayer);
  });

  $('portsTable').querySelector('tbody').innerHTML = data.ports.map(p => `
    <tr>
      <td>${p.year}</td>
      <td>${p.port_name}</td>
      <td>${p.state}</td>
      <td>${fmt.format(Math.round(p.value || 0))}</td>
    </tr>
  `).join('');
}

async function boot() {
  const [options, summary] = await Promise.all([
    getJSON('/api/options/'),
    getJSON('/api/summary/')
  ]);

  upsertOptions($('state'), options.states, 'All states');
  upsertOptions($('year'), options.years, 'All years', '');

  setSummary(summary);
  await loadCountries();
  await loadMonthlyRisk();
  await loadHotspots();
}

$('refresh').addEventListener('click', async () => {
  selectedCountry = null;
  await loadCountries();
  await loadMonthlyRisk();
  await loadHotspots();
});

boot().catch(err => {
  console.error(err);
  alert('Dashboard failed to load. Did you run migrations and import_data? See README.md.');
});
