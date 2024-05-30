from osgeo import ogr, gdal, osr
from time import time

def main(shapefile):

#making the shapefile as an object.
    input_shp = ogr.Open(shapefile)

    #getting layer information of shapefile.
    shp_layer = input_shp.GetLayer()

    #pixel_size determines the size of the new raster.
    #pixel_size is proportional to size of shapefile.
    pixel_size = 0.1

    #get extent values to set size of output raster.
    x_min, x_max, y_min, y_max = shp_layer.GetExtent()
    print(x_min, x_max)
    print(y_min, y_max)
    #calculate size/resolution of the raster.
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)

    #get GeoTiff driver by 
    image_type = 'GTiff'
    driver = gdal.GetDriverByName(image_type)

    #passing the filename, x and y direction resolution, no. of bands, new raster.
    new_raster = driver.Create(output_raster, x_res, y_res, 1, gdal.GDT_Byte)

    #transforms between pixel raster space to projection coordinate space.
    new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))

    #get required raster band.
    band = new_raster.GetRasterBand(1)
    #assign no data value to empty cells.
    no_data_value = -9999
    band.SetNoDataValue(no_data_value)
    band.FlushCache()

    #main conversion method
    gdal.RasterizeLayer(new_raster, [1], shp_layer, burn_values=[255], options=['ALL_TOUCHED=FALSE'])
    #adding a spatial reference
    new_rasterSRS = osr.SpatialReference()
    new_rasterSRS.ImportFromEPSG(5179)
    new_raster.SetProjection(new_rasterSRS.ExportToWkt())
    return gdal.Open(output_raster)

def crop_tiff(output_shape, input_shape):
    upper_left_x = 921817.4965573151
    upper_left_y = 1956010.095815223
    lower_right_x = 941536.3622601214
    lower_right_y = 1950959.0655774989
    window = (upper_left_x,lower_right_y,upper_left_x+100,lower_right_y+100)
    gdal.Translate(output_shape, input_shape, projWin = window)
    
def crop_shape(output_tif, input_tif):
    upper_left_x = 921817.4965573151
    upper_left_y = 1956010.095815223
    lower_right_x = 941536.3622601214
    lower_right_y = 1950959.0655774989
    window = (upper_left_x,lower_right_y,upper_left_x+100,lower_right_y+100)
    gdal.VectorTranslate(output_tif, input_tif, spatFilter = window)

if __name__ == "__main__":
    output_raster = "test.tiff"
    cropped_raster = "test_2.tiff"
    # main("/home/chanuk/workspaces/mcar_ws/src/localization/gps-world-tf/map_data/B2_SURFACELINEMARK.shp")
    before = time()
    # crop_tiff(cropped_raster,output_raster)
    crop_shape("test_3.shp", "/home/chanuk/workspaces/mcar_ws/src/localization/gps-world-tf/map_data/B2_SURFACELINEMARK.shp")
    print (time() - before)