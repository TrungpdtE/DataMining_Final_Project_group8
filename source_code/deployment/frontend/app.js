const { useEffect, useMemo, useRef, useState } = React;

const yen = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "JPY",
  maximumFractionDigits: 0,
});

const FIELD_LABELS = {
  Type: "Loại bất động sản",
  Prefecture: "Tỉnh / đô thị",
  Location: "Khu vực",
  price: "Giá thực tế",
  predicted_price_yen: "Giá dự đoán",
  Year: "Năm",
  Quarter: "Quý",
  area: "Diện tích đất m2",
  floor_area: "Diện tích sàn m2",
  frontage: "Mặt tiền",
  coverage: "Mật độ xây dựng %",
  FloorAreaRatio: "Hệ số sử dụng đất",
  house_age: "Tuổi nhà",
  AverageTimeToStation: "Phút tới ga",
  MunicipalityCategory: "Loại đô thị",
  density: "Mật độ dân số",
  Migration: "Di cư",
  Distance_to_designated_city: "Khoảng cách tới đô thị chỉ định",
  dist_to_tokyo_km: "Cách Tokyo km",
  dist_to_osaka_km: "Cách Osaka km",
  dist_to_nagoya_km: "Cách Nagoya km",
  dist_to_fukuoka_km: "Cách Fukuoka km",
  dist_to_sapporo_km: "Cách Sapporo km",
  dist_to_nearest_major_center_km: "Cách trung tâm lớn gần nhất km",
  location_cluster: "Cụm vị trí",
  Prefecture_target_mean: "Giá TB theo tỉnh",
  Location_target_mean: "Giá TB theo khu vực",
  Close_to_Tokyo: "Gần Tokyo",
  Close_to_greater_Tokyo_area: "Gần vùng Tokyo mở rộng",
  Close_to_designated_city_flag: "Gần đô thị chỉ định",
  gdp_growth: "Tăng trưởng GDP %",
  interest_rate: "Lãi suất %",
  inflation_rate: "Lạm phát %",
  population_growth: "Tăng dân số %",
  housing_price_index: "Chỉ số giá nhà",
  last_year_prefecture_avg_price: "Giá TB tỉnh năm trước",
  last_year_prefecture_tx_count: "Số giao dịch tỉnh năm trước",
  prefecture_price_growth_1y: "Tăng giá tỉnh 1 năm",
  prefecture_price_growth_3y: "Tăng giá tỉnh 3 năm",
  last_year_location_avg_price: "Giá TB khu vực năm trước",
  location_price_growth_1y: "Tăng giá khu vực 1 năm",
  last_year_global_avg_price: "Giá TB toàn quốc năm trước",
  global_price_growth_1y: "Tăng giá toàn quốc 1 năm",
  global_price_growth_3y: "Tăng giá toàn quốc 3 năm",
  nearest_major_center: "Trung tâm lớn gần nhất",
  Nearest_designated_city: "Đô thị chỉ định gần nhất",
  split_role: "Nhóm dữ liệu",
};

function labelFor(key) {
  return FIELD_LABELS[key] || key.replaceAll("_", " ");
}

function roleLabel(role) {
  return {
    train: "Huấn luyện",
    test: "Kiểm định 2024",
    forecast: "Dự phóng",
    future_observed: "Quan sát mới",
  }[role] || role || "-";
}

// 47 prefecture colors — each gets a distinct pastel hue
const PREFECTURE_COLORS = [
  "#FFB3B3","#FFD9B3","#FFFAB3","#D4FFB3","#B3FFD1","#B3FFF5","#B3E4FF",
  "#B3C4FF","#D1B3FF","#F5B3FF","#FFB3E4","#FFB3C4","#FFCBB3","#F5FFB3",
  "#C4FFB3","#B3FFE4","#B3F0FF","#B3D1FF","#C4B3FF","#FFB3F5","#FFB3D1",
  "#FFD4B3","#EFFFB3","#B3FFB8","#B3FFD9","#B3F5FF","#B3CCFF","#D9B3FF",
  "#FFB3F0","#FFB3CC","#FFDDB3","#E8FFB3","#B3FFC4","#B3FFE8","#B3EEFF",
  "#B3D9FF","#C9B3FF","#F0B3FF","#FFB3E8","#FFB3D9","#FFE8B3","#F0FFB3",
  "#B3FFCC","#B3FFEC","#B3F0FF","#B3D4FF","#CCB3FF",
];

function prefectureColor(name, alpha = 0.35) {
  // stable hash so same name always gets same color
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) & 0xffff;
  const hex = PREFECTURE_COLORS[h % PREFECTURE_COLORS.length];
  // convert #RRGGBB → rgba
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function colorFor(value) {
  const million = (Number(value) || 0) / 1_000_000;
  if (million >= 65) return "#d95f0e";
  if (million >= 35) return "#fec44f";
  if (million >= 18) return "#41ab5d";
  return "#2b8cbe";
}

function strokeFor(role) {
  if (role === "test") return "#d7191c";
  if (role === "forecast") return "#7b2cbf";
  if (role === "future_observed") return "#fdae61";
  return "#2166ac";
}

function popupHtml(props) {
  const actual = props.price ? yen.format(props.price) : "Không có giá thực tế";
  const pred = props.predicted_price_yen ? yen.format(props.predicted_price_yen) : "Chưa có dự đoán";
  const predClass = priceDeltaClass(props.price, props.predicted_price_yen);
  return `
    <div class="popup-title">${props.Location || "Không rõ khu vực"}</div>
    <div>${props.Prefecture || ""}</div>
    <div>Năm: <b>${props.Year || "-"}</b> Q${props.Quarter || "-"}</div>
    <div>Đất: <b>${props.area || "-"} m2</b>, sàn: <b>${props.floor_area || "-"} m2</b></div>
    <div>Tới ga: <b>${props.AverageTimeToStation || "-"} phút</b></div>
    <div>Nhóm: <b>${roleLabel(props.split_role)}</b></div>
    <hr />
    <div class="popup-price-grid">
      <div><span>Giá thực tế</span><b>${actual}</b></div>
      <div><span>Giá dự đoán</span><b class="${predClass}">${pred}</b></div>
    </div>
  `;
}

function priceDeltaClass(actual, predicted) {
  const a = Number(actual);
  const p = Number(predicted);
  if (!Number.isFinite(a) || !Number.isFinite(p)) return "";
  return p >= a ? "price-up" : "price-down";
}

function formatDetailValue(value, key = "") {
  if (value === null || value === undefined || value === "") return "-";
  if (key === "price" || key === "predicted_price_yen" || key.endsWith("_price") || key.endsWith("_avg_price")) {
    return yen.format(value);
  }
  if (typeof value === "number") {
    if (Math.abs(value) >= 1000) return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
    return Number.isInteger(value) ? String(value) : value.toLocaleString("en-US", { maximumFractionDigits: 3 });
  }
  return String(value);
}

function DetailRow({ label, value, valueClass = "" }) {
  return React.createElement(
    "div", { className: "receipt-row" },
    React.createElement("span", null, labelFor(label)),
    React.createElement("b", { className: valueClass }, formatDetailValue(value, label))
  );
}

function featureValue(feature, mode) {
  const props = feature.properties || {};
  return mode === "historical" ? props.price : props.predicted_price_yen || props.price;
}

function TimelineChart({ points }) {
  if (!points.length) {
    return React.createElement("div", { className: "empty-chart" }, "Chạy dự đoán để xem mô phỏng theo năm.");
  }
  const values = points.map((p) => p.estimated_price_million_yen);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const polyline = points
    .map((p, i) => {
      const x = 18 + (i * 264) / Math.max(1, points.length - 1);
      const y = 118 - ((p.estimated_price_million_yen - min) / span) * 88;
      return `${x},${y}`;
    })
    .join(" ");
  return React.createElement(
    "div", { className: "chart-wrap" },
    React.createElement(
      "svg", { viewBox: "0 0 300 140", role: "img" },
      React.createElement("polyline", { points: polyline, fill: "none", stroke: "#16685a", strokeWidth: "3" }),
      points.map((p, i) => {
        const x = 18 + (i * 264) / Math.max(1, points.length - 1);
        const y = 118 - ((p.estimated_price_million_yen - min) / span) * 88;
        return React.createElement("circle", { key: p.year, cx: x, cy: y, r: "4", fill: "#c7532f" });
      })
    ),
    React.createElement(
      "div", { className: "chart-labels" },
      points.map((p) =>
        React.createElement("span", { key: p.year }, `${p.year}: ${p.estimated_price_million_yen.toFixed(1)}M`)
      )
    )
  );
}

// ── Japan prefecture boundaries (simplified GeoJSON polygons) ──────────────
// We fetch from a public CDN so no backend needed.
function ReceiptPanel({ estimate, timeline, selectedContext, selectedRecord, form }) {
  if (!estimate && !selectedRecord) {
    return React.createElement(
      "div", { className: "receipt-empty" },
      "Bấm một điểm trên đất liền Nhật Bản trong AI Ước tính hoặc chọn một marker dữ liệu trong Bản đồ thị trường."
    );
  }

  if (selectedRecord) {
    const props = selectedRecord.properties || {};
    const detailKeys = Object.keys(props)
      .filter((key) => !["price", "predicted_price_yen"].includes(key))
      .sort((a, b) => a.localeCompare(b));
    return React.createElement(
      "div", { className: "receipt" },
      React.createElement("div", { className: "receipt-kicker" }, "Điểm dữ liệu thị trường"),
      React.createElement("h3", null, props.Location || "Không rõ khu vực"),
      React.createElement("div", { className: "receipt-price-grid" },
        React.createElement("div", null,
          React.createElement("span", null, "Giá thực tế"),
          React.createElement("strong", null, props.price ? yen.format(props.price) : "-")
        ),
        React.createElement("div", null,
          React.createElement("span", null, "Giá dự đoán"),
          React.createElement("strong", { className: priceDeltaClass(props.price, props.predicted_price_yen) },
            props.predicted_price_yen ? yen.format(props.predicted_price_yen) : "-"
          )
        )
      ),
      React.createElement("div", { className: "receipt-lines" },
        detailKeys.map((key) => React.createElement(DetailRow, { key, label: key, value: props[key] }))
      )
    );
  }

  return React.createElement(
    "div", { className: "receipt" },
    React.createElement("div", { className: "receipt-kicker" }, "Phiếu ước tính AI"),
    React.createElement("h3", null, `${estimate.location || form.location || "-"}, ${estimate.prefecture || form.prefecture || "-"}`),
    React.createElement("div", { className: "receipt-price-grid single" },
      React.createElement("div", null,
        React.createElement("span", null, "Giá dự đoán"),
        React.createElement("strong", null, yen.format(estimate.estimated_price_yen))
      )
    ),
    React.createElement("div", { className: "receipt-lines" },
      React.createElement(DetailRow, { label: "Year", value: estimate.year }),
      React.createElement(DetailRow, { label: "Prefecture", value: estimate.prefecture || form.prefecture }),
      React.createElement(DetailRow, { label: "Location", value: estimate.location || form.location }),
      React.createElement(DetailRow, { label: "area", value: form.area }),
      React.createElement(DetailRow, { label: "floor_area", value: form.floor_area }),
      React.createElement(DetailRow, { label: "frontage", value: form.frontage }),
      React.createElement(DetailRow, { label: "coverage", value: form.coverage }),
      React.createElement(DetailRow, { label: "FloorAreaRatio", value: form.floor_area_ratio }),
      React.createElement(DetailRow, { label: "Type", value: form.property_type }),
      React.createElement(DetailRow, { label: "Năm xây dựng", value: form.construction_year }),
      React.createElement(DetailRow, { label: "AverageTimeToStation", value: form.average_time_to_station }),
      React.createElement(DetailRow, { label: "gdp_growth", value: form.gdp_growth }),
      React.createElement(DetailRow, { label: "interest_rate", value: form.interest_rate }),
      React.createElement(DetailRow, { label: "inflation_rate", value: form.inflation_rate }),
      React.createElement(DetailRow, { label: "housing_price_index", value: form.housing_price_index })
    ),
    selectedContext && React.createElement(
      "div", { className: "receipt-note" },
      React.createElement("b", null, "Ga gần nhất"),
      React.createElement("span", null,
        `${selectedContext.nearest_station.name} - ${selectedContext.nearest_station.distance_km.toFixed(2)} km - ${selectedContext.nearest_station.walking_minutes} min`
      ),
      React.createElement("small", null, `Ngữ cảnh dữ liệu gần nhất: ${selectedContext.location}, ${selectedContext.prefecture}`)
    ),
    React.createElement("h2", { className: "subhead" }, "Mô phỏng theo năm"),
    React.createElement(TimelineChart, { points: timeline })
  );
}

const JAPAN_GEOJSON_URL =
  "https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson";

function App() {
  const mapRef        = useRef(null);
  const layerRef      = useRef(null);
  const heatLayerRef  = useRef(null);
  const stationLayerRef = useRef(null);
  const prefLayerRef  = useRef(null);   // prefecture polygons
  const selectedLineRef = useRef(null);
  const markerRef     = useRef(null);
  const formRef       = useRef(null);
  const yearRef       = useRef(2024);
  const activeTabRef  = useRef("estimator");

  const [year, setYear]         = useState(2024);
  const [mode, setMode]         = useState("predicted");
  const [activeTab, setActiveTab] = useState("estimator");
  const [layerMode, setLayerMode] = useState("clusters");
  const [showStations, setShowStations] = useState(true);
  const [showPrefectureColors, setShowPrefectureColors] = useState(true);
  const [options, setOptions]   = useState({ prefectures: [], locations: [], years: [2005, 2024] });
  const [optionsLoaded, setOptionsLoaded] = useState(false);
  const [macroLoaded,   setMacroLoaded]   = useState(false);
  const [busy,    setBusy]      = useState(false);
  const [status,  setStatus]    = useState("Sẵn sàng");
  const [estimate, setEstimate] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [geojson,  setGeojson]  = useState({ type: "FeatureCollection", features: [] });
  const [stations, setStations] = useState({ type: "FeatureCollection", features: [] });
  const [selectedContext, setSelectedContext] = useState(null);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [mapFilter, setMapFilter] = useState({ prefecture: "ALL", location: "ALL", limit: 5000 });
  const [form, setForm] = useState({
    prefecture: "Tokyo",
    location: "",
    area: 120,
    floor_area: 90,
    frontage: 8,
    coverage: 60,
    floor_area_ratio: 200,
    property_type: "",
    construction_year: 2010,
    average_time_to_station: 15,
    gdp_growth: 0,
    interest_rate: 0,
    inflation_rate: 0,
    population_growth: 0,
    housing_price_index: 100,
    latitude: null,
    longitude: null,
  });
  formRef.current = form;
  yearRef.current = year;
  activeTabRef.current = activeTab;

  const mapLocations = useMemo(() =>
    options.locations.filter((item) => mapFilter.prefecture === "ALL" || item.Prefecture === mapFilter.prefecture),
    [options.locations, mapFilter.prefecture]
  );
  const predictionLocations = useMemo(() =>
    options.locations.filter((item) => !form.prefecture || item.Prefecture === form.prefecture),
    [options.locations, form.prefecture]
  );
  const topFeatures = useMemo(() => geojson.features.slice(0, 250), [geojson]);

  // ── 1. Init Leaflet map ────────────────────────────────────────────────────
  useEffect(() => {
    const mapEl = document.getElementById("map");
    if (!mapEl) return;

    function initMap() {
      const japanBounds = L.latLngBounds([24.0, 122.0], [46.5, 153.5]);

      const map = L.map("map", {
        preferCanvas: true,
        minZoom: 4,
        zoomAnimation: false,
        fadeAnimation: false,
        markerZoomAnimation: false,
        inertia: false,
      });

      L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        {
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
          subdomains: "abcd",
          maxZoom: 20,
        }
      ).addTo(map);

      map.whenReady(() => {
        map.invalidateSize({ pan: false });
        map.fitBounds(japanBounds, { padding: [20, 20], animate: false });
      });

      const ro = new ResizeObserver(() => map.invalidateSize({ pan: false }));
      ro.observe(mapEl);
      window.addEventListener("resize", () => map.invalidateSize({ pan: false }));

      mapRef.current = map;

      // ── Load Japan prefecture GeoJSON for coloring ──
      fetch(JAPAN_GEOJSON_URL)
        .then((r) => r.json())
        .then((gj) => {
          const layer = L.geoJSON(gj, {
            style: (feature) => {
              const name = feature.properties.nam_ja || feature.properties.name || "";
              return {
                fillColor:   prefectureColor(name, 0.12),
                fillOpacity: 1,
                color:       "#576b64",
                weight:      0.8,
                opacity:     0.28,
              };
            },
            onEachFeature: (feature, lyr) => {
              const name = feature.properties.nam_ja || feature.properties.name || "";
              lyr.bindTooltip(name, { sticky: true, className: "pref-tooltip" });
            },
          });
          prefLayerRef.current = layer;
          layer.addTo(map);
        })
        .catch(() => console.warn("Không tải được ranh giới tỉnh Nhật Bản"));

      // ── Map click handler ──
      map.on("click", (event) => {
        if (activeTabRef.current !== "estimator") return;
        const lat = Number(event.latlng.lat.toFixed(6));
        const lng = Number(event.latlng.lng.toFixed(6));
        setStatus(`Đã chọn ${lat.toFixed(4)}, ${lng.toFixed(4)}. Đang dựng ngữ cảnh vị trí.`);
        fetch(`/location/context?latitude=${lat}&longitude=${lng}`)
          .then((res) => res.json())
          .then((context) => {
            if (!context?.prefecture || Number(context.context_distance_km) > 35) {
              setStatus("Điểm này quá xa dữ liệu bất động sản Nhật Bản. Hãy chọn trên đất liền, gần thành phố hoặc quận.");
              setSelectedContext(null);
              return;
            }
            const overrides = {
              latitude: lat,
              longitude: lng,
              prefecture: context.prefecture,
              location: context.location,
              average_time_to_station: Math.round(
                context.suggested_average_time_to_station ?? formRef.current.average_time_to_station
              ),
              frontage: Number(context.suggested_frontage ?? formRef.current.frontage ?? 8).toFixed(1),
              coverage: Number(context.suggested_coverage ?? formRef.current.coverage ?? 60).toFixed(0),
              floor_area_ratio: Number(
                context.suggested_floor_area_ratio ?? formRef.current.floor_area_ratio ?? 200
              ).toFixed(0),
              property_type: context.suggested_type || formRef.current.property_type,
            };
            setForm((prev) => ({ ...prev, ...overrides }));
            setSelectedContext(context);
            setSelectedRecord(null);
            setActiveTab("estimator");
            drawSelectedContext(lat, lng, context);
            setStatus(
              `Ngữ cảnh: ${context.location}, ${context.prefecture}. ` +
              `Ga gần nhất: ${context.nearest_station.name} ` +
              `(${context.nearest_station.distance_km.toFixed(2)} km).`
            );
            runPredict(overrides);
          })
          .catch(() => {
            setStatus("Không tạo được ngữ cảnh đất liền đáng tin cậy cho điểm này.");
          });
      });

      // ── Fetch app options ──
      fetch("/options")
        .then((res) => res.json())
        .then((data) => {
          setOptions(data);
          const first = data.prefectures?.[0]?.name || "Tokyo";
          setForm((prev) => ({ ...prev, prefecture: first, property_type: data.types?.[0] || "" }));
          setYear(data.years?.[1] || 2024);
          setOptionsLoaded(true);
        })
        .catch(() => { setOptionsLoaded(true); setStatus("Không tải được danh sách tùy chọn"); });

      fetch("/macro/latest?year=2026")
        .then((res) => res.json())
        .then((data) => {
          setForm((prev) => ({
            ...prev,
            gdp_growth:        Number(data.gdp_growth        ?? prev.gdp_growth).toFixed(2),
            interest_rate:     Number(data.interest_rate     ?? prev.interest_rate).toFixed(2),
            inflation_rate:    Number(data.inflation_rate    ?? prev.inflation_rate).toFixed(2),
            population_growth: Number(data.population_growth ?? prev.population_growth).toFixed(2),
            housing_price_index: Number(data.housing_price_index ?? prev.housing_price_index).toFixed(1),
          }));
          setMacroLoaded(true);
        })
        .catch(() => setMacroLoaded(true));

      return () => {
        ro.disconnect();
        map.remove();
      };
    }

    if (mapEl.offsetWidth > 0 && mapEl.offsetHeight > 0) {
      return initMap();
    } else {
      let cleanup;
      const raf = requestAnimationFrame(() => { cleanup = initMap(); });
      return () => { cancelAnimationFrame(raf); if (cleanup) cleanup(); };
    }
  }, []);

  // ── 2. Load ALL stations for all Japan once, keep in state ────────────────
  useEffect(() => {
    // Try all-Japan endpoint first; fall back to per-prefecture concat
    const endpoints = ["/stations?prefecture=ALL&limit=0"];
    let done = false;
    const tryNext = (idx) => {
      if (idx >= endpoints.length || done) return;
      fetch(endpoints[idx])
        .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
        .then((data) => {
          if (data.features && data.features.length > 0) {
            done = true;
            setStations(data);
          } else {
            tryNext(idx + 1);
          }
        })
        .catch(() => tryNext(idx + 1));
    };
    tryNext(0);
  }, []);

  // ── 3. Render station markers ──────────────────────────────────────────────
  useEffect(() => {
    if (!mapRef.current) return;
    if (stationLayerRef.current) {
      mapRef.current.removeLayer(stationLayerRef.current);
      stationLayerRef.current = null;
    }
    if (!showStations || !stations.features.length) return;

    const stationIcon = L.divIcon({
      className: "station-icon",
      html: "S",
      iconSize: [16, 16],
      iconAnchor: [8, 8],
    });
    const group = L.markerClusterGroup
      ? L.markerClusterGroup({ chunkedLoading: true, maxClusterRadius: 40, disableClusteringAtZoom: 13 })
      : L.layerGroup();

    stations.features.forEach((feature) => {
      const coords = feature.geometry.coordinates;
      const props  = feature.properties || {};
      group.addLayer(
        L.marker([coords[1], coords[0]], { icon: stationIcon, bubblingMouseEvents: false }).bindPopup(
          `<b>${props.name || "Ga tàu"}</b><br/>${props.prefecture || props.line || "Nhật Bản"}`
        )
      );
    });
    stationLayerRef.current = group.addTo(mapRef.current);
  }, [stations, activeTab, showStations]);

  // ── 4. Tab-switch: refocus map ─────────────────────────────────────────────
  useEffect(() => {
    if (!mapRef.current) return;
    const map = mapRef.current;
    const japanBounds = L.latLngBounds([24.0, 122.0], [46.5, 153.5]);

    const refocus = () => {
      map.invalidateSize({ pan: false });
      if (activeTab === "history") {
        map.fitBounds(japanBounds, { padding: [20, 20], animate: false });
      } else if (!formRef.current?.latitude && !formRef.current?.longitude) {
        // No property selected yet — show all Japan
        map.fitBounds(japanBounds, { padding: [20, 20], animate: false });
      }
      // If a property is selected, keep current view
    };
    const t = setTimeout(refocus, 50);
    return () => clearTimeout(t);
  }, [activeTab]);

  // ── 5. Prefecture polygon visibility by tab ────────────────────────────────
  useEffect(() => {
    if (!mapRef.current || !prefLayerRef.current) return;
    if (showPrefectureColors) {
      if (!mapRef.current.hasLayer(prefLayerRef.current)) {
        prefLayerRef.current.addTo(mapRef.current);
      }
    } else {
      if (mapRef.current.hasLayer(prefLayerRef.current)) {
        mapRef.current.removeLayer(prefLayerRef.current);
      }
    }
  }, [showPrefectureColors, activeTab]);

  // ── 6. History map data ────────────────────────────────────────────────────
  useEffect(() => {
    if (!mapRef.current || !optionsLoaded || !macroLoaded) return;
    if (activeTab !== "history") {
      setGeojson({ type: "FeatureCollection", features: [] });
      if (layerRef.current)     mapRef.current.removeLayer(layerRef.current);
      if (heatLayerRef.current) mapRef.current.removeLayer(heatLayerRef.current);
      if (activeTab === "estimator") {
        setStatus("Chế độ ước tính: bấm lên bản đồ để đặt bất động sản.");
      }
      return;
    }
    setBusy(true);
    const controller = new AbortController();
    const historyYear = Math.min(year, options.years[1] || 2024);
    const params = new URLSearchParams({
      year, limit: mapFilter.limit, prefecture: mapFilter.prefecture, location: mapFilter.location,
      gdp_growth: form.gdp_growth, interest_rate: form.interest_rate,
      inflation_rate: form.inflation_rate, population_growth: form.population_growth,
      housing_price_index: form.housing_price_index,
    });
    const historicalParams = new URLSearchParams({
      year: historyYear, limit: mapFilter.limit,
      prefecture: mapFilter.prefecture, location: mapFilter.location,
    });
    const endpoint = mode === "historical" ? `/map?${historicalParams}` : `/map/predictions?${params}`;
    fetch(endpoint, { signal: controller.signal })
      .then((r) => r.json())
      .then((data) => {
        setGeojson(data);
        const label = mapFilter.location !== "ALL" ? mapFilter.location
          : mapFilter.prefecture !== "ALL" ? mapFilter.prefecture : "Toàn Nhật Bản";
        setStatus(`Đã tải ${data.features.length.toLocaleString()} điểm cho ${label}`);
      })
      .catch((err) => { if (err.name !== "AbortError") setStatus("Không tải được dữ liệu bản đồ"); })
      .finally(() => setBusy(false));
    return () => controller.abort();
  }, [year, mode, activeTab, optionsLoaded, macroLoaded,
      mapFilter.prefecture, mapFilter.location, mapFilter.limit,
      form.gdp_growth, form.interest_rate, form.inflation_rate,
      form.population_growth, form.housing_price_index]);

  // ── 7. Render history layer ────────────────────────────────────────────────
  useEffect(() => {
    if (!mapRef.current) return;
    if (activeTab !== "history") {
      if (layerRef.current)     mapRef.current.removeLayer(layerRef.current);
      if (heatLayerRef.current) mapRef.current.removeLayer(heatLayerRef.current);
      return;
    }
    if (layerRef.current)     mapRef.current.removeLayer(layerRef.current);
    if (heatLayerRef.current) mapRef.current.removeLayer(heatLayerRef.current);

    const heatPoints = geojson.features.map((f) => {
      const coords = f.geometry.coordinates;
      const intensity = Math.max(0.1, Math.min(1, (Number(featureValue(f, mode)) || 0) / 80_000_000));
      return [coords[1], coords[0], intensity];
    });

    if (layerMode === "heatmap" || layerMode === "both") {
      heatLayerRef.current = L.heatLayer(heatPoints, {
        radius: 22, blur: 18, maxZoom: 12,
        gradient: { 0.15: "#2b8cbe", 0.45: "#41ab5d", 0.7: "#fec44f", 1: "#d95f0e" },
      }).addTo(mapRef.current);
    }
    if (layerMode === "clusters" || layerMode === "both") {
      const group = L.markerClusterGroup
        ? L.markerClusterGroup({ chunkedLoading: true, maxClusterRadius: 45, disableClusteringAtZoom: 13 })
        : L.layerGroup();
      geojson.features.forEach((f) => {
        const props = f.properties || {};
        const coords = f.geometry.coordinates;
        const marker = L.circleMarker([coords[1], coords[0]], {
          radius: layerMode === "both" ? 3.2 : 4.5,
          stroke: true,
          color: strokeFor(props.split_role),
          weight: props.split_role === "test" ? 2 : 1.1,
          fillOpacity: layerMode === "both" ? 0.52 : 0.72,
          fillColor: colorFor(featureValue(f, mode)),
          bubblingMouseEvents: false,
        }).bindPopup(popupHtml(props));
        marker.on("click", () => {
          setSelectedRecord({ properties: props, coordinates: coords });
          setSelectedContext(null);
          setStatus(`Đã chọn điểm dữ liệu: ${props.Location || "Không rõ"}, ${props.Prefecture || ""}`);
        });
        group.addLayer(marker);
      });
      layerRef.current = group.addTo(mapRef.current);
    }
  }, [geojson, layerMode, mode, activeTab]);

  // ── Helpers ────────────────────────────────────────────────────────────────
  function update(key, value) { setForm((prev) => ({ ...prev, [key]: value })); }

  function updateMapFilter(key, value) {
    setMapFilter((prev) => ({ ...prev, [key]: value, location: key === "prefecture" ? "ALL" : value }));
  }

  function drawSelectedContext(lat, lng, context) {
    if (!mapRef.current || !context?.nearest_station) return;
    if (markerRef.current)      mapRef.current.removeLayer(markerRef.current);
    if (selectedLineRef.current) mapRef.current.removeLayer(selectedLineRef.current);

    const userIcon = L.divIcon({ className: "picked-icon",   html: "●", iconSize: [24,24], iconAnchor: [12,12] });
    const stIcon   = L.divIcon({ className: "nearest-station-icon", html: "S", iconSize: [22,22], iconAnchor: [11,11] });
    const station  = context.nearest_station;

    const group = L.layerGroup([
      L.marker([lat, lng], { icon: userIcon, bubblingMouseEvents: false }).bindPopup(`<b>Bất động sản đã chọn</b><br/>${lat.toFixed(5)}, ${lng.toFixed(5)}`),
      L.marker([station.latitude, station.longitude], { icon: stIcon, bubblingMouseEvents: false }).bindPopup(
        `<b>${station.name}</b><br/>${station.distance_km.toFixed(2)} km · ${station.walking_minutes} min`
      ),
    ]).addTo(mapRef.current);

    selectedLineRef.current = L.polyline(
      [[lat, lng], [station.latitude, station.longitude]],
      { color: "#c7532f", weight: 3, dashArray: "8 6" }
    ).addTo(mapRef.current);

    markerRef.current = group;
    mapRef.current.fitBounds(
      L.latLngBounds([[lat, lng], [station.latitude, station.longitude]]),
      { padding: [80, 80], maxZoom: 15 }
    );
  }

  function zoomToFeature(feature) {
    const coords = feature.geometry.coordinates;
    const latlng = [coords[1], coords[0]];
    mapRef.current.setView(latlng, 13);
    if (markerRef.current) mapRef.current.removeLayer(markerRef.current);
    markerRef.current = L.marker(latlng).addTo(mapRef.current);
    markerRef.current.bindPopup(popupHtml(feature.properties || {})).openPopup();
    setSelectedRecord({ properties: feature.properties || {}, coordinates: coords });
    setSelectedContext(null);
  }

  function runPredict(overrides = {}) {
    setBusy(true);
    const payload = { ...(formRef.current || form), ...overrides, year: yearRef.current, quarter: 2 };
    fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => {
        setEstimate(data);
        setSelectedRecord(null);
        setActiveTab("receipt");
        setStatus("Đã cập nhật dự đoán");
        runTimeline(payload);
        if (data.latitude && data.longitude && mapRef.current) {
          const latlng = [data.latitude, data.longitude];
          mapRef.current.setView(latlng, 12);
          if (markerRef.current) mapRef.current.removeLayer(markerRef.current);
          markerRef.current = L.marker(latlng).addTo(mapRef.current);
          markerRef.current
            .bindPopup(`<b>${data.location}</b><br/>${data.prefecture}<br/>${yen.format(data.estimated_price_yen)}`)
            .openPopup();
        }
      })
      .catch(() => setStatus("Dự đoán thất bại"))
      .finally(() => setBusy(false));
  }

  function runTimeline(payload = { ...(formRef.current || form), year: yearRef.current, quarter: 2 }) {
    fetch("/predict/timeline", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => setTimeline(data.points || []))
      .catch(() => setTimeline([]));
  }

  function retrain() {
    fetch("/train", { method: "POST" })
      .then((r) => r.json())
      .then((data) => setStatus(data.message || "Đã đưa tác vụ huấn luyện vào hàng đợi"))
      .catch(() => setStatus("Không thể bắt đầu huấn luyện"));
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return React.createElement(
    "div", { className: "shell" },

    // ── Sidebar ──
    React.createElement(
      "aside", { className: "sidebar" },

      React.createElement(
        "div", { className: "brand" },
        React.createElement(
          "div", null,
          React.createElement("h1", null, "Dự đoán giá nhà Nhật Bản"),
          React.createElement("p",  null, "Bấm bản đồ để tạo ước tính mới hoặc kiểm tra dữ liệu thị trường.")
        ),
        React.createElement("span", { className: "badge" }, mode === "historical" ? "Thực tế" : "Mô hình")
      ),

      React.createElement(
        "div", { className: "tabs" },
        React.createElement("button", { className: activeTab === "estimator" ? "active" : "", onClick: () => setActiveTab("estimator") }, "AI Ước tính"),
        React.createElement("button", { className: activeTab === "history"   ? "active" : "", onClick: () => setActiveTab("history")   }, "Thị trường"),
        React.createElement("button", { className: activeTab === "receipt"   ? "active" : "", onClick: () => setActiveTab("receipt")   }, "Phiếu")
      ),
      React.createElement(
        "div", { className: "toolbar compact layer-toggles" },
        React.createElement("button", { className: `btn ${showStations ? "" : "secondary"}`, onClick: () => setShowStations((value) => !value) }, "Ga tàu"),
        React.createElement("button", { className: `btn ${showPrefectureColors ? "" : "secondary"}`, onClick: () => setShowPrefectureColors((value) => !value) }, "Tô tỉnh")
      ),

      // ── History tab controls ──
      activeTab === "history" && React.createElement(
        "div", { className: "section" },
        React.createElement("h2", null, "Dữ liệu bản đồ"),
        React.createElement("label", null,
          `Năm ${year}`,
          React.createElement("input", { type: "range", min: options.years[0] || 2005, max: 2026, value: year, onChange: (e) => setYear(Number(e.target.value)) })
        ),
        React.createElement("label", null, "Phạm vi dữ liệu",
          React.createElement("select", { value: mapFilter.prefecture, onChange: (e) => updateMapFilter("prefecture", e.target.value) },
            React.createElement("option", { value: "ALL" }, "Toàn Nhật Bản"),
            options.prefectures.map((p) => React.createElement("option", { key: p.name, value: p.name }, `${p.name} (${p.count})`))
          )
        ),
        React.createElement("label", null, "Thành phố / quận",
          React.createElement("select", { value: mapFilter.location, onChange: (e) => updateMapFilter("location", e.target.value) },
            React.createElement("option", { value: "ALL" }, "Tất cả khu vực"),
            mapLocations.map((p) => React.createElement("option", { key: `${p.Prefecture}-${p.Location}`, value: p.Location }, `${p.Location} (${p.count})`))
          )
        ),
        React.createElement("label", null, "Số điểm tối đa",
          React.createElement("select", { value: mapFilter.limit, onChange: (e) => updateMapFilter("limit", Number(e.target.value)) },
            [
              { value: 2000, label: "2,000" },
              { value: 5000, label: "5,000" },
              { value: 10000, label: "10,000" },
              { value: 20000, label: "20,000" },
              { value: 50000, label: "50,000" },
              { value: 100000, label: "100,000" },
              { value: 0, label: "All" },
            ].map((item) => React.createElement("option", { key: item.value, value: item.value }, item.label))
          )
        ),
        React.createElement("div", { className: "toolbar compact" },
          React.createElement("button", { className: `btn ${mode === "predicted"  ? "" : "secondary"}`, onClick: () => setMode("predicted")  }, "Dự đoán"),
          React.createElement("button", { className: `btn ${mode === "historical" ? "" : "secondary"}`, onClick: () => setMode("historical") }, "Lịch sử")
        ),
        React.createElement("div", { className: "toolbar compact" },
          React.createElement("button", { className: `btn ${layerMode === "clusters" ? "" : "secondary"}`, onClick: () => setLayerMode("clusters") }, "Cụm"),
          React.createElement("button", { className: `btn ${layerMode === "heatmap"  ? "" : "secondary"}`, onClick: () => setLayerMode("heatmap")  }, "Bản đồ nhiệt"),
          React.createElement("button", { className: `btn ${layerMode === "both"     ? "" : "secondary"}`, onClick: () => setLayerMode("both")     }, "Cả hai")
        ),
        React.createElement("div", { className: "status" }, status)
      ),

      activeTab === "history" && React.createElement(
        "div", { className: "section" },
        React.createElement("h2", null, "Điểm dữ liệu"),
        React.createElement(
          "div", { className: "data-list" },
          topFeatures.map((feature, index) => {
            const props = feature.properties || {};
            return React.createElement(
              "button", { key: index, className: "data-row", onClick: () => zoomToFeature(feature) },
              React.createElement("span", { className: `dot ${props.split_role || "train"}` }),
              React.createElement("span", null, props.Location || "Không rõ"),
              React.createElement("small", null, `${props.Prefecture || ""} · ${yen.format(featureValue(feature, mode) || 0)}`)
            );
          })
        )
      ),

      // ── Estimator tab controls ──
      activeTab === "estimator" && React.createElement(
        "div", { className: "section" },
        React.createElement("h2", null, "AI ước tính giá"),
        React.createElement("div", { className: "status" }, "Bấm một điểm trên đất liền Nhật Bản để tự điền vị trí, rồi chỉnh thông tin nếu cần."),
        React.createElement("label", null, "Tỉnh / đô thị",
          React.createElement("select", { value: form.prefecture, onChange: (e) => update("prefecture", e.target.value) },
            options.prefectures.map((p) => React.createElement("option", { key: p.name, value: p.name }, `${p.name} (${p.count})`))
          )
        ),
        React.createElement("label", null, "Thành phố / quận",
          React.createElement("select", { value: form.location, onChange: (e) => update("location", e.target.value) },
            React.createElement("option", { value: "" }, "Tự chọn theo tỉnh"),
            predictionLocations.map((p) => React.createElement("option", { key: `${p.Prefecture}-${p.Location}`, value: p.Location }, `${p.Location} (${p.count})`))
          )
        ),
        React.createElement("div", { className: "row" },
          React.createElement("label", null, "Diện tích đất",   React.createElement("input", { type: "number", value: form.area,       onChange: (e) => update("area",       Number(e.target.value)) })),
          React.createElement("label", null, "Diện tích sàn",  React.createElement("input", { type: "number", value: form.floor_area,  onChange: (e) => update("floor_area",  Number(e.target.value)) }))
        ),
        React.createElement("div", { className: "row" },
          React.createElement("label", null, "Mặt tiền",    React.createElement("input", { type: "number", value: form.frontage,    onChange: (e) => update("frontage",    Number(e.target.value)) })),
          React.createElement("label", null, "Mật độ xây dựng %",  React.createElement("input", { type: "number", value: form.coverage,    onChange: (e) => update("coverage",    Number(e.target.value)) }))
        ),
        React.createElement("div", { className: "row" },
          React.createElement("label", null, "Hệ số sử dụng đất", React.createElement("input", { type: "number", value: form.floor_area_ratio, onChange: (e) => update("floor_area_ratio", Number(e.target.value)) })),
          React.createElement("label", null, "Loại bất động sản",
            React.createElement("select", { value: form.property_type, onChange: (e) => update("property_type", e.target.value) },
              options.types?.map((t) => React.createElement("option", { key: t, value: t }, t))
            )
          )
        ),
        React.createElement("div", { className: "row" },
          React.createElement("label", null, "Năm xây dựng",   React.createElement("input", { type: "number", value: form.construction_year,       onChange: (e) => update("construction_year",       Number(e.target.value)) })),
          React.createElement("label", null, "Phút tới ga",  React.createElement("input", { type: "number", value: form.average_time_to_station, onChange: (e) => update("average_time_to_station", Number(e.target.value)) }))
        ),
        form.latitude && React.createElement("div", { className: "status" },
          `Điểm đã chọn: ${Number(form.latitude).toFixed(4)}, ${Number(form.longitude).toFixed(4)}`
        ),
        selectedContext && React.createElement(
          "div", { className: "context-box" },
          React.createElement("b", null, "Ga gần nhất"),
          React.createElement("span", null,
            `${selectedContext.nearest_station.name} · ${selectedContext.nearest_station.distance_km.toFixed(2)} km · ${selectedContext.nearest_station.walking_minutes} min`
          ),
          React.createElement("small", null, `Ngữ cảnh dữ liệu gần nhất: ${selectedContext.location}, ${selectedContext.prefecture}`)
        )
      ),

      activeTab === "estimator" && React.createElement(
        "div", { className: "section" },
        React.createElement("h2", null, "Kịch bản Nhật Bản 2026"),
        React.createElement("div", { className: "row" },
          React.createElement("label", null, "Tăng trưởng GDP %",  React.createElement("input", { type: "number", step: "0.1", value: form.gdp_growth,     onChange: (e) => update("gdp_growth",     Number(e.target.value)) })),
          React.createElement("label", null, "Lãi suất %",    React.createElement("input", { type: "number", step: "0.1", value: form.interest_rate,   onChange: (e) => update("interest_rate",   Number(e.target.value)) }))
        ),
        React.createElement("div", { className: "row" },
          React.createElement("label", null, "Lạm phát %",   React.createElement("input", { type: "number", step: "0.1", value: form.inflation_rate,  onChange: (e) => update("inflation_rate",  Number(e.target.value)) })),
          React.createElement("label", null, "Chỉ số giá nhà", React.createElement("input", { type: "number", step: "1",   value: form.housing_price_index, onChange: (e) => update("housing_price_index", Number(e.target.value)) }))
        ),
        React.createElement("div", { className: "toolbar" },
          React.createElement("button", { className: "btn", disabled: busy, onClick: () => runPredict() }, "Dự đoán"),
          React.createElement("button", { className: "btn warn", onClick: retrain }, "Huấn luyện lại")
        ),
        estimate && React.createElement(
          "div", { className: "metric compact-result" },
          React.createElement("span",   null, `${estimate.location}, ${estimate.prefecture}`),
          React.createElement("strong", null, yen.format(estimate.estimated_price_yen))
        )
      ),

      (activeTab === "receipt" || (activeTab === "history" && selectedRecord)) && React.createElement(
        "div", { className: "section receipt-section" },
        React.createElement("h2", null, "Kết quả đã chọn"),
        React.createElement(ReceiptPanel, {
          estimate,
          timeline,
          selectedContext,
          selectedRecord,
          form,
        })
      )
    ),

    // ── Map panel ──
    React.createElement(
      "main", { className: "map-wrap" },
      React.createElement(
        "div", { className: "map-head" },
        React.createElement("div", { className: "map-chip" },
          stations.features.length > 0
            ? `Đã tải ${stations.features.length.toLocaleString()} ga · Bấm bản đồ để dự đoán`
            : "Đang tải ga tàu... · Bấm bản đồ để dự đoán"
        ),
        React.createElement(
          "div", { className: "legend" },
          React.createElement("b", null, "Màu giá"),
          React.createElement("div", { className: "scale" },
            React.createElement("i"), React.createElement("i"), React.createElement("i"), React.createElement("i")
          ),
          React.createElement("small", null, "<18tr, 18-35tr, 35-65tr, 65tr+ JPY"),
          React.createElement("div", { className: "split-legend" },
            React.createElement("span", { className: "train"    }, "Huấn luyện"),
            React.createElement("span", { className: "test"     }, "Kiểm định 2024"),
            React.createElement("span", { className: "forecast" }, "Dự phóng")
          )
        )
      ),
      React.createElement("div", { id: "map" })
    )
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(App));
