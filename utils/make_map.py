import folium as f

colors = ["red", "blue", "black", "green", "purple", "orange", "darkred", "darkblue", "darkgreen", "gray", "darkpurple"]

def make_map(map_df, cluster_field, save=False, map_file_name="", with_points=False):
  m = f.Map()
  m.fit_bounds([[map_df.Event_Lat.min(), map_df.Event_Long.min()],
                [map_df.Event_Lat.max(), map_df.Event_Long.max()]])

  i = 0

  for cluster in map_df[cluster_field].unique():
    df_temp_clust = map_df[map_df[cluster_field] == cluster]
    color = colors[i]
    if i < len(colors) - 1:
      i += 1
    else:
      i = 0
    fg = f.FeatureGroup(name=str(cluster), overlay=True, control=True)
    for leg_id in df_temp_clust["leg_id"].unique():
      df_temp = df_temp_clust[df_temp_clust['leg_id'] == leg_id]
      df_temp = df_temp.sort_values(by=['Event_Dttm'])
      fg.add_child(f.PolyLine(list(zip(df_temp.Event_Lat, df_temp.Event_Long)),
                              color=color, popup=str(leg_id)))
      if with_points:
        for lat, lon, leg_id in list(zip(df_temp.Event_Lat, df_temp.Event_Long, df_temp.leg_id)):
          f.Circle((lat, lon), color=color, radius=10).add_to(f.FeatureGroup(name=cluster).add_to(m))
    m.add_child(fg)
  
  f.LayerControl(collapsed=True).add_to(m)
  if save:
    print("Saving into {}".format(map_file_name))
    m.save(outfile=str(map_file_name) + ".html")

  f.LayerControl(collapsed=False).add_to(m)
  return m

def plot_map(coords, fmap, _name, _color, with_points=False, label=""):
  fg = f.FeatureGroup(name=_name, overlay=True, control=True)
  fg.add_child(f.PolyLine(coords, color=_color, popup=label))
  fmap.add_child(fg)
  fmap.add_child(fg)

def plot_points(coords, fmap, _color):
  for lat, lon in coords:
    f.Circle((lat, lon), color=_color, radius=10).add_to(f.FeatureGroup(name="points").add_to(fmap))

def make_map_with_line_segments(lines_M, lines_N, with_points=False, save=False, map_file_name="", distance_label=""):
  m = f.Map()
  xm = [m["p1"][0] for m in lines_M] + [m["p2"][0] for m in lines_M]
  ym = [m["p1"][1] for m in lines_M] + [m["p2"][1] for m in lines_M]
  xn = [n["p1"][0] for n in lines_N] + [n["p2"][0] for n in lines_N]
  yn = [n["p1"][1] for n in lines_N] + [n["p2"][1] for n in lines_N]

  x_min = min(xm + xn)
  x_max = max(xm + xn)
  y_min = min(ym + yn)
  y_max = max(ym + yn)

  m.fit_bounds([[x_min, y_min], [x_max, y_max]])
  
  # plot lines_M
  lines_M = sorted(lines_M, key=lambda x: x["dttm"])
  lm_coords = []
  for line in lines_M:
    coords = [line["p1"]] + [line["p2"]]
    lm_coords += coords
  plot_map(lm_coords, m, "line_M", colors[0], with_points, distance_label)

  # plot lines_N
  lines_N = sorted(lines_N, key=lambda x: x["dttm"])
  ln_coords = []
  for line in lines_N:
    coords = [line["p1"]] + [line["p2"]]
    ln_coords += coords
  plot_map(ln_coords, m, "line_N", colors[1], with_points, distance_label)
  
  if with_points:
    plot_points(lm_coords + ln_coords, m, colors[2])
  f.LayerControl(collapsed=True).add_to(m)

  if save:
    print("Saving into {}".format(map_file_name))
    m.save(outfile=str(map_file_name) + ".html")

  f.LayerControl(collapsed=False).add_to(m)
  return m
