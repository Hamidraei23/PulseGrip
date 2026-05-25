/****************************************************************************
** Copyright (C) 2020 MikroElektronika d.o.o.
** Contact: https://www.mikroe.com/contact
**
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and associated documentation files (the "Software"), to deal
** in the Software without restriction, including without limitation the rights
** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
** copies of the Software, and to permit persons to whom the Software is
** furnished to do so, subject to the following conditions:
** The above copyright notice and this permission notice shall be
** included in all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
** EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
** OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
** IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
** DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
** OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
**  USE OR OTHER DEALINGS IN THE SOFTWARE.
****************************************************************************/

/*!
 * @file c3dhall11.c
 * @brief 3D Hall 11 Click Driver.
 */

#include <Arduino.h>
#include <Wire.h>
#include "c3dhall11.h"

C3dhall11::C3dhall11()
{
  i2c_address = C3DHALL11_DEVICE_ADDRESS;
  offset_data.x_axis = 0;
  offset_data.y_axis = 0;
  offset_data.z_axis = 0;
  offset_data.magnitude = 0;
}
C3dhall11::~C3dhall11()
{
}
void C3dhall11::setI2CInstance(TwoWire* i2c){
  this->WireInstance = i2c;
}

void C3dhall11::set_address(uint8_t new_address)
{
    
    write_register ( C3DHALL11_I2C_ADDRESS_REGISTER, new_address<<1 | I2C_ADDRESS_UPDATE_EN );
    i2c_address = new_address;
        
}

void C3dhall11::offsetCalibration(){
  c3dhall11_data_t sensor_data;
  
  for(uint8_t i = 0; i<10;i++){
    read_data_calibration(&sensor_data);
    offset_data.x_axis += sensor_data.x_axis;
    offset_data.y_axis += sensor_data.y_axis;
    offset_data.z_axis += sensor_data.z_axis;
    offset_data.magnitude += sensor_data.magnitude;
    delay(5);
  }
  offset_data.x_axis /= 10;
  offset_data.y_axis /= 10;
  offset_data.z_axis /= 10;
  offset_data.magnitude /= 10;

}

int C3dhall11::default_cfg ()
{
    int error_flag = C3DHALL11_OK;
    if ( C3DHALL11_ERROR == check_communication() )
    {
        return C3DHALL11_ERROR;
    }
    write_register ( C3DHALL11_REG_DEVICE_CONFIG_1,C3DHALL11_MAG_TEMPCO_0p12 | C3DHALL11_CONV_AVG_32X | 
                                                                                 C3DHALL11_I2C_RD_STANDARD );
    write_register ( C3DHALL11_REG_SENSOR_CONFIG_1, C3DHALL11_MAG_CH_EN_ENABLE_XYZ );
    write_register ( C3DHALL11_REG_SENSOR_CONFIG_2, C3DHALL11_ANGLE_EN_NO_ANGLE | 
                                                                                 C3DHALL11_X_Y_RANGE_40mT | 
                                                                                 C3DHALL11_Z_RANGE_80mT );
    write_register (  C3DHALL11_REG_T_CONFIG, C3DHALL11_T_CH_EN_DISABLE );
    write_register (  C3DHALL11_REG_INT_CONFIG_1, C3DHALL11_MASK_INTB_DISABLE );
    write_register (  C3DHALL11_REG_DEVICE_CONFIG_2, C3DHALL11_OPERATING_MODE_CONTINUOUS );
    
    return error_flag;
}

void C3dhall11::generic_write ( uint8_t reg, uint8_t *data_in, uint8_t len )
{
    uint8_t data_buf[ 256 ] = { 0 };
    
    this->WireInstance->beginTransmission(i2c_address);
    this->WireInstance->write(reg);

    for ( uint8_t cnt = 0; cnt < len; cnt++ )
    {
        this->WireInstance->write(*data_in); 
    } 
    
    this->WireInstance->endTransmission(); 

}

void C3dhall11::generic_read ( uint8_t reg, uint8_t *data_out, uint8_t len )
{
     
    this->WireInstance->beginTransmission(i2c_address);
    this->WireInstance->write(reg); 
    this->WireInstance->endTransmission();

    this->WireInstance->requestFrom(i2c_address, len);
    if(len<=this->WireInstance->available()){
        for (int i=0; i<len; i++)
        data_out[i]=this->WireInstance->read();
    }


}

void C3dhall11::write_register ( uint8_t reg, uint8_t data_in )
{
    generic_write( reg, &data_in, 1 );
}

void C3dhall11::read_register ( uint8_t reg, uint8_t *data_out )
{
    generic_read( reg, data_out, 1 );
}


int C3dhall11::check_communication ()
{
    uint8_t data_buf[ 3 ] = { 0 };
    generic_read(C3DHALL11_REG_DEVICE_ID, data_buf, 3 );
    {
        if ( ( C3DHALL11_DEVICE_ID == data_buf[ 0 ] ) && 
             ( C3DHALL11_MANUFACTURER_ID_LSB == data_buf[ 1 ] ) &&
             ( C3DHALL11_MANUFACTURER_ID_MSB == data_buf[ 2 ] ) )
        {
            return C3DHALL11_OK;
        }
    }
    return C3DHALL11_ERROR;
}
uint8_t C3dhall11::read_data_calibration ( c3dhall11_data_t *data_out )
{
    uint8_t data_buf[ 12 ] = { 0 };

    generic_read ( C3DHALL11_REG_T_MSB_RESULT, data_buf, 12 );
    if ( ( data_buf[ 8 ] & C3DHALL11_CONV_STATUS_DATA_READY ) )
    {
        uint8_t sensor_config[ 2 ] = { 0 };
        generic_read ( C3DHALL11_REG_SENSOR_CONFIG_1, sensor_config, 2 );
        if ( sensor_config[ 0 ] & C3DHALL11_MAG_CH_EN_BIT_MASK )
        {
            data_out->magnitude = data_buf[ 11 ];
        }

        data_out->x_axis = ( float )(int16_t)((data_buf[ 2 ] << 8) | data_buf[ 3 ]);
        data_out->y_axis = ( float )(int16_t)((data_buf[ 4 ] << 8) | data_buf[ 5 ]);
        data_out->z_axis = ( float )(int16_t)((data_buf[ 6 ] << 8) | data_buf[ 7 ]);
        if ( sensor_config[ 1 ] & C3DHALL11_X_Y_RANGE_80mT )
        {
            data_out->x_axis /= C3DHALL11_XYZ_SENSITIVITY_80mT;
            data_out->y_axis /= C3DHALL11_XYZ_SENSITIVITY_80mT; 
        }
        else
        {
            data_out->x_axis /= C3DHALL11_XYZ_SENSITIVITY_40mT;
            data_out->y_axis /= C3DHALL11_XYZ_SENSITIVITY_40mT; 
        }
        if ( sensor_config[ 1 ] & C3DHALL11_Z_RANGE_80mT )
        {
            data_out->z_axis /= C3DHALL11_XYZ_SENSITIVITY_80mT;
        }
        else
        {
            data_out->z_axis /= C3DHALL11_XYZ_SENSITIVITY_40mT;
        }
        return C3DHALL11_OK;
    }
    return C3DHALL11_ERROR;
}

int C3dhall11::read_data ( c3dhall11_data_t *data_out )
{
    uint8_t data_buf[ 12 ] = { 0 };

    generic_read ( C3DHALL11_REG_T_MSB_RESULT, data_buf, 12 );
    if ( ( data_buf[ 8 ] & C3DHALL11_CONV_STATUS_DATA_READY ) )
    {
        uint8_t sensor_config[ 2 ] = { 0 };
        generic_read ( C3DHALL11_REG_SENSOR_CONFIG_1, sensor_config, 2 );
        if ( sensor_config[ 0 ] & C3DHALL11_MAG_CH_EN_BIT_MASK )
        {
            data_out->magnitude = data_buf[ 11 ]-offset_data.magnitude;
        }
        if ( sensor_config[ 1 ] & C3DHALL11_ANGLE_EN_BIT_MASK )
        {
            data_out->angle = (float)( uint16_t )((data_buf[ 9 ] << 8 ) | data_buf[ 10 ] ) / C3DHALL11_ANGLE_RESOLUTION;
        }
        uint8_t temp_config = 0;
        read_register (C3DHALL11_REG_T_CONFIG, &temp_config );
        if ( temp_config & C3DHALL11_T_CH_EN_ENABLE )
        {
            data_out->temperature = C3DHALL11_TEMP_SENS_T0 + 
                                    ((float)( uint16_t ) ((data_buf[ 0 ] << 8 ) | data_buf[ 1 ] ) - C3DHALL11_TEMP_ADC_T0 ) 
                                    / C3DHALL11_TEMP_ADC_RESOLUTION;
        }
        data_out->x_axis = ( float )(int16_t)((data_buf[ 2 ] << 8) | data_buf[ 3 ]);
        data_out->y_axis = ( float )(int16_t)((data_buf[ 4 ] << 8) | data_buf[ 5 ]);
        data_out->z_axis = ( float )(int16_t)((data_buf[ 6 ] << 8) | data_buf[ 7 ]);
        if ( sensor_config[ 1 ] & C3DHALL11_X_Y_RANGE_80mT )
        {
            data_out->x_axis /= C3DHALL11_XYZ_SENSITIVITY_80mT;
            data_out->x_axis -= offset_data.x_axis;
            data_out->y_axis /= C3DHALL11_XYZ_SENSITIVITY_80mT; 
            data_out->y_axis -= offset_data.y_axis;
        }
        else
        {
            data_out->x_axis /= C3DHALL11_XYZ_SENSITIVITY_40mT;
            data_out->x_axis -= offset_data.x_axis;
            data_out->y_axis /= C3DHALL11_XYZ_SENSITIVITY_40mT;
            data_out->y_axis -= offset_data.y_axis;
        }
        if ( sensor_config[ 1 ] & C3DHALL11_Z_RANGE_80mT )
        {
            data_out->z_axis /= C3DHALL11_XYZ_SENSITIVITY_80mT;
            data_out->z_axis -= offset_data.z_axis;
        }
        else
        {
            data_out->z_axis /= C3DHALL11_XYZ_SENSITIVITY_40mT;
            data_out->z_axis -= offset_data.z_axis;
        }
        return C3DHALL11_OK;
    }
    return C3DHALL11_ERROR;
}

// ------------------------------------------------------------------------- END
